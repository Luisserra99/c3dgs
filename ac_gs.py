import numpy as np
from io import BytesIO

from tqdm import tqdm

from arithmeticcoding import BitOutputStream, \
    ArithmeticEncoder, FlatFrequencyTable, SimpleFrequencyTable 
    
    
from arithmeticcoding import ArithmeticDecoder, BitInputStream


def initialize_feature_rest_freq_models(feature_shape):

    freq_models = []

    # initialize a flat frequency model for each SH coefficient (for RGB channels separately)
    for i in range(feature_shape[0]):
        for j in range(feature_shape[1]):
            freq_models.append(FlatFrequencyTable(257))  # uniform over [0..256]

    return freq_models

def encode_rgb_features_ac(compressed_data, key, fname):
    print(f"starting to encode {key} with arithmetic coding...")
    bitstream = open(fname, "wb")

    data_sh_ac = compressed_data[key].astype(np.int16) + 128  # shift to unsigned
    data_sh_ac = data_sh_ac.astype(np.uint16)

    bitout = BitOutputStream(bitstream)

    freq_models = initialize_feature_rest_freq_models(data_sh_ac.shape[1:])

    encoder = ArithmeticEncoder(32, bitout)

    # Use tqdm to track progress on large arrays.
    for value in tqdm(data_sh_ac, desc=f"AC encode {key}", unit="gauss"):
        for sh_par in range(value.shape[0]):
            for rgb_par in range(value.shape[1]):
                idx = sh_par * value.shape[1] + rgb_par
                encoder.write(freq_models[idx], value[sh_par, rgb_par])
                freq_models[idx].increment(value[sh_par, rgb_par])

    encoder.write(freq_models[0], 256)  # EOF symbol
    encoder.finish()

    compressed_data.pop(key)  # save this separately as a binary file

    bitout.close()


def encode_feature_rest(compressed_data, fname):
    encode_rgb_features_ac(compressed_data, "features_rest", fname)


def encode_int8_array_ac(compressed_data, key, fname):
    print(f"starting to encode {key} with arithmetic coding...")
    bitstream = open(fname, "wb")

    data_sh_ac = compressed_data[key].astype(np.int16) + 128  # shift to unsigned
    data_sh_ac = data_sh_ac.astype(np.uint16)

    bitout = BitOutputStream(bitstream)
    freq_model = FlatFrequencyTable(257)
    encoder = ArithmeticEncoder(32, bitout)

    for value in tqdm(data_sh_ac.flatten(), desc=f"AC encode {key}", unit="sym"):
        encoder.write(freq_model, int(value))

    encoder.write(freq_model, 256)  # EOF symbol
    encoder.finish()

    compressed_data.pop(key)
    bitout.close()

def encode_compress_features(compressed_data, fname):
    print('starting to encode feature indices with arithmetic coding...')
    data_vqi_features = compressed_data["feature_indices"]

    # Determine the symbol range dynamically based on the data.
    # We reserve one extra symbol for EOF.
    max_symbol = int(data_vqi_features.max())
    eof_symbol = max_symbol + 1

    # bitstream = BytesIO()
    bitstream = open(fname, "wb")

    bitout = BitOutputStream(bitstream)
    freq_models = FlatFrequencyTable(eof_symbol + 1)
    encoder = ArithmeticEncoder(32, bitout)

    # Use tqdm to track progress on large arrays.
    for value in tqdm(data_vqi_features, desc="AC encode feature_indices", unit="sym"):
        encoder.write(freq_models, int(value))

    encoder.write(freq_models, eof_symbol)  # EOF symbol
    encoder.finish()

    #compressed_data["feature_indices_vq"] = bitstream.getvalue()
    compressed_data.pop("feature_indices")  # we will save this separately as a binary file, since it is large and has a simple distribution that compresses well with AC

    bitout.close()

def encode_compress_gaussians(compressed_data, fname):
    print('starting to encode gaussian indices with arithmetic coding...')
    data_vqi_gaussian = compressed_data["gaussian_indices"]

    # Determine the symbol range dynamically based on the data.
    # We reserve one extra symbol for EOF.
    max_symbol = int(data_vqi_gaussian.max())
    eof_symbol = max_symbol + 1

    # bitstream = BytesIO()
    bitstream = open(fname, "wb")

    bitout = BitOutputStream(bitstream)
    # Use a flat (uniform) model to keep encoding speed predictable for large symbol ranges.
    freq_models = FlatFrequencyTable(eof_symbol + 1)

    encoder = ArithmeticEncoder(32, bitout)

    # Use tqdm to track progress on large arrays.
    for value in tqdm(data_vqi_gaussian, desc="AC encode gaussian_indices", unit="sym"):
        encoder.write(freq_models, int(value))

    encoder.write(freq_models, eof_symbol)  # EOF symbol
    encoder.finish()

    #compressed_data["gaussian_indices_vq"] = bitstream.getvalue()
    compressed_data.pop("gaussian_indices")  # we will save this separately as a binary file, since it is large and has a simple distribution that compresses well with AC
    bitout.close()
    
def encoder_gs_params():
     
    # a = in_memory_bitstream.getvalue()
    # a = in_memory_bitstream.getvalue()
    
    #data = np.load('raw_decomp.npz', allow_pickle=True)
    _raw_data = np.load('raw_decomp_unc.npz', allow_pickle=True)
    
    compressed_data = dict()
    for data_key in _raw_data.files:
        compressed_data[data_key] = _raw_data[data_key].copy()
        
    
    ##################################################
    ### Code SH AC coefficients
    ##################################################
    
    encode_feature_rest(compressed_data, "features_rest.bin")
    
    ##################################################
    ### VQ indices - feature_indices_vq
    ##################################################
    encode_compress_features(compressed_data, "feature_indices.bin")
    
    
    ##################################################
    ### VQ indices - gaussian_indices_vq
    ##################################################

    encode_compress_gaussians(compressed_data, "gaussian_indices.bin")


    np.savez_compressed("compressed_ac.npz", **compressed_data)
    
    
    # https://www.nayuki.io/page/reference-arithmetic-coding
    
def decode_rgb_features_ac(fname, feature_shape=(15, 3), num_gaussians=None):
    """Decode an RGB feature bitstream produced by encode_rgb_features_ac.

    Returns a numpy array of shape (N, *feature_shape) with dtype int8.

    Args:
        fname: path to bitstream
        feature_shape: per-gaussian feature shape (typically (15, 3) or (3, 1))
        num_gaussians: optional expected number of gaussians; when provided,
            decoding stops after this many gaussians even if EOF is not reached.
    """

    bitstream = open(fname, "rb")
    bitin = BitInputStream(bitstream)
    decoder = ArithmeticDecoder(32, bitin)

    freq_models = initialize_feature_rest_freq_models(feature_shape)

    features = []
    nb_gaussians = 0
    have_seen_eof = False

    while not have_seen_eof:
        nb_gaussians += 1
        sh = np.zeros(feature_shape, dtype=np.uint8)

        for sh_par in range(feature_shape[0]):
            for rgb_par in range(feature_shape[1]):
                idx = sh_par * feature_shape[1] + rgb_par
                symbol = decoder.read(freq_models[idx])

                if symbol == 256:  # EOF symbol
                    have_seen_eof = True
                    break

                sh[sh_par, rgb_par] = symbol

            if have_seen_eof:
                break

        if not have_seen_eof:
            features.append(sh)
            print(f"Decoded gaussian {nb_gaussians}...", end="\r")

        if num_gaussians is not None and nb_gaussians >= num_gaussians:
            break

    bitin.close()

    features = (np.stack(features, dtype=np.int16) - 128).astype(np.int8)  # shift back to signed
    return features


def decode_features_rest_ac(fname, feature_shape=(15, 3), num_gaussians=None):
    return decode_rgb_features_ac(fname, feature_shape=feature_shape, num_gaussians=num_gaussians)


def decode_int8_array_ac(fname, shape=None, num_values=None):
    """Decode an int8 bitstream produced by encode_int8_array_ac.

    Args:
        fname: path to bitstream file.
        shape: optional target shape for the decoded array.
        num_values: optional number of values to decode; overrides shape if set.

    Returns:
        np.ndarray of decoded int8 values.
    """

    bitstream = open(fname, "rb")
    bitin = BitInputStream(bitstream)
    decoder = ArithmeticDecoder(32, bitin)
    freq_model = FlatFrequencyTable(257)

    values = []
    target_len = None
    if num_values is not None:
        target_len = int(num_values)
    elif shape is not None:
        target_len = int(np.prod(shape))

    while True:
        symbol = decoder.read(freq_model)
        if symbol == 256:
            break
        values.append(symbol)
        if target_len is not None and len(values) >= target_len:
            break

    bitin.close()

    out = (np.array(values, dtype=np.int16) - 128).astype(np.int8)
    if shape is not None:
        out = out.reshape(shape)
    return out


def decode_vq_indices_ac(fname, max_value):
    """Decode a VQ index bitstream created by encode_compress_*.

    Args:
        fname: path to bitstream file.
        max_value: maximum symbol value present in the original data.

    Returns:
        np.ndarray of decoded integer symbols.
    """

    eof_symbol = max_value + 1
    bitstream = open(fname, "rb")
    bitin = BitInputStream(bitstream)
    decoder = ArithmeticDecoder(32, bitin)

    freq_model = FlatFrequencyTable(eof_symbol + 1)

    values = []
    while True:
        symbol = decoder.read(freq_model)
        if symbol == eof_symbol:
            break
        values.append(symbol)

    bitin.close()
    return np.array(values, dtype=np.int32)


def decoder_gs_params():
    
    decode_features_rest_ac("features_rest.bin")
    
    
  
if __name__ == "__main__":
    
    # example encoding features_rest, gaussian_indices_vq and feature_indices_vq with arithmetic coding
    encoder_gs_params()
    
    # exmple decoding features_rest
    decoder_gs_params()
