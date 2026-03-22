import os
import numpy as np
from io import BytesIO

from tqdm import tqdm

from arithmeticcoding import BitOutputStream, \
    ArithmeticEncoder, FlatFrequencyTable, SimpleFrequencyTable 
    
    
from arithmeticcoding import ArithmeticDecoder, BitInputStream


def initialize_feature_rest_freq_models(feature_shape):

    default_freq = FlatFrequencyTable(257)
    freq_models = []

    # initialize a flat frequency model for each SH coefficient (for RGB channels separately)
    for i in range(feature_shape[0]):
        for j in range(feature_shape[1]):
            freq_models.append(SimpleFrequencyTable(default_freq))  # uniform over [0..256]

    return freq_models

def encode_feature_rest(compressed_data, key, fname):
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
                symbol = int(value[sh_par, rgb_par])
                encoder.write(freq_models[idx], symbol)
                freq_models[idx].increment(symbol)

    encoder.write(freq_models[0], 256)  # EOF symbol
    encoder.finish()

    compressed_data.pop(key)  # save this separately as a binary file

    bitout.close()



def encode_int8_array_ac(compressed_data, key, fname):
    print(f"starting to encode {key} with arithmetic coding...")
    bitstream = open(fname, "wb")

    data_sh_ac = compressed_data[key].astype(np.int16) + 128  # shift to unsigned
    data_sh_ac = data_sh_ac.astype(np.uint16)

    bitout = BitOutputStream(bitstream)
    max_symbol = int(data_sh_ac.max())
    eof_symbol = max_symbol + 1
    freq_model = SimpleFrequencyTable(FlatFrequencyTable(eof_symbol + 1))
    encoder = ArithmeticEncoder(32, bitout)

    for value in tqdm(data_sh_ac, desc=f"AC encode {key}", unit="gauss"):
        for symbol in value.flatten():
            encoder.write(freq_model, int(symbol))
            freq_model.increment(int(symbol))  

    encoder.write(freq_model, eof_symbol)  
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
    freq_models = SimpleFrequencyTable(FlatFrequencyTable(eof_symbol + 1)) 
    encoder = ArithmeticEncoder(32, bitout)

    # Use tqdm to track progress on large arrays.
    for value in tqdm(data_vqi_features, desc="AC encode feature_indices", unit="sym"):
        encoder.write(freq_models, int(value))
        freq_models.increment(int(value))  

    encoder.write(freq_models, eof_symbol)  
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
    # FIX: use adaptive SimpleFrequencyTable instead of static flat model for better compression
    freq_models = SimpleFrequencyTable(FlatFrequencyTable(eof_symbol + 1))

    encoder = ArithmeticEncoder(32, bitout)

    # Use tqdm to track progress on large arrays.
    for value in tqdm(data_vqi_gaussian, desc="AC encode gaussian_indices", unit="sym"):
        encoder.write(freq_models, int(value))
        freq_models.increment(int(value))  # FIX: update model after each symbol so codebook skew is exploited

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

    # tqdm: use num_gaussians as total when known; otherwise show running count
    with tqdm(total=num_gaussians, desc=f"AC decode {os.path.basename(fname)}", unit="gauss") as pbar:
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
                    freq_models[idx].increment(symbol)

                if have_seen_eof:
                    break

            if not have_seen_eof:
                features.append(sh)
                pbar.update(1)  # replaced print(...\r) with tqdm update

            if num_gaussians is not None and nb_gaussians >= num_gaussians:
                break

    bitin.close()

    features = (np.stack(features, dtype=np.int16) - 128).astype(np.int8)  # shift back to signed
    return features


def decode_features_rest_ac(fname, feature_shape=(15, 3), num_gaussians=None):
    return decode_rgb_features_ac(fname, feature_shape=feature_shape, num_gaussians=num_gaussians)


def decode_int8_array_ac(fname, shape=None, num_values=None, eof_symbol=256):
    """Decode an int8 bitstream produced by encode_int8_array_ac.

    Args:
        fname: path to bitstream file.
        shape: optional target shape for the decoded array.
        num_values: optional number of values to decode; overrides shape if set.
        eof_symbol: must match the value used by the encoder (default 256).
            The encoder sets eof_symbol = max(shifted_data) + 1, which is only 256
            when all 256 int8 values are present.  Pass the value from metadata.json
            to guarantee the frequency-table sizes match exactly.

    Returns:
        np.ndarray of decoded int8 values.
    """

    bitstream = open(fname, "rb")
    bitin = BitInputStream(bitstream)
    decoder = ArithmeticDecoder(32, bitin)
    # Table size must match the encoder: FlatFrequencyTable(eof_symbol + 1)
    freq_model = SimpleFrequencyTable(FlatFrequencyTable(eof_symbol + 1))  # was hardcoded 257

    values = []
    target_len = None
    if num_values is not None:
        target_len = int(num_values)
    elif shape is not None:
        target_len = int(np.prod(shape))

    # tqdm: show total symbols when known, otherwise show running count
    with tqdm(total=target_len, desc=f"AC decode {os.path.basename(fname)}", unit="sym") as pbar:
        while True:
            symbol = decoder.read(freq_model)
            if symbol == eof_symbol:  # was hardcoded 256; now uses the matched eof_symbol
                break
            values.append(symbol)
            freq_model.increment(symbol)
            pbar.update(1)  # advance progress bar by one decoded symbol
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

    freq_model = SimpleFrequencyTable(FlatFrequencyTable(eof_symbol + 1)) 

    values = []
    # total is unknown at decode time (no shape metadata for VQ indices); show running count
    with tqdm(desc=f"AC decode {os.path.basename(fname)}", unit="sym") as pbar:
        while True:
            symbol = decoder.read(freq_model)
            if symbol == eof_symbol:
                break
            values.append(symbol)
            freq_model.increment(symbol)
            pbar.update(1)  # added: advance progress bar per decoded symbol

    bitin.close()
    return np.array(values, dtype=np.int32)


def decoder_gs_params():
    
    decode_features_rest_ac("features_rest.bin")
    
    
  
if __name__ == "__main__":
    
    # example encoding features_rest, gaussian_indices_vq and feature_indices_vq with arithmetic coding
    encoder_gs_params()
    
    # exmple decoding features_rest
    decoder_gs_params()
