import numpy as np
from matplotlib import pyplot as plt

from io import BytesIO

from ac.arithmetic_coding import BitOutputStream, \
    ArithmeticEncoder, FlatFrequencyTable, SimpleFrequencyTable 
    
    
from ac.arithmetic_coding import ArithmeticDecoder, BitInputStream


def initialize_feature_rest_freq_models(feature_shape):
    
    default_freq = FlatFrequencyTable(257)
    freq_models = []
    
    # initialize a frequency model for each SH coefficient (for RGB channels separately)
    for i in range(feature_shape[0]):
        for j in range(feature_shape[1]):
            freq_models.append( SimpleFrequencyTable(default_freq) )  # start with a uniform distribution
            
    return freq_models

def encode_feature_rest(compressed_data, fname):
    
    #bitstream = BytesIO()
    bitstream = open(fname, "wb")
     
    data_sh_ac = compressed_data["features_rest"] + 128 # shift to unsigned
    data_sh_ac = data_sh_ac.astype(np.uint8)
    
    # bitout = BitOutputStream(open("dev.bin", "wb"))
    bitout = BitOutputStream(bitstream)
    
    freq_models = initialize_feature_rest_freq_models(data_sh_ac.shape[1:])
    
    
    encoder = ArithmeticEncoder(32, bitout) 
    for value in data_sh_ac:
        
        for sh_par in range(value.shape[0]):
            for rgb_par in range(value.shape[1]):
                idx = sh_par * value.shape[1] + rgb_par
                encoder.write(freq_models[idx], value[sh_par, rgb_par])
                freq_models[idx].increment(value[sh_par, rgb_par])
        
    encoder.write(freq_models[0], 256)  # EOF symbol
    encoder.finish()
    
    bitout.end()
    
    # compressed_data["features_rest"] = bitstream.getvalue()
    compressed_data.pop("features_rest") # we will save this separately as a binary file, since it is large and has a simple distribution that compresses well with AC
    
    bitout.close()

def compress_vq_features(compressed_data, fname):
    
    data_vqi_features = compressed_data["feature_indices_vq"]
    
    # bitstream = BytesIO()
    bitstream = open(fname, "wb")
    
    bitout = BitOutputStream(bitstream)
    default_freq = FlatFrequencyTable(2**12+1)
    freq_models = SimpleFrequencyTable(default_freq)
    
    encoder = ArithmeticEncoder(32, bitout)
    
    for value in data_vqi_features:
        
        encoder.write(freq_models, value)
        freq_models.increment(value)
    
    encoder.write(freq_models, 2**12)  # EOF symbol
    encoder.finish()
   
    bitout.end()
        
    #compressed_data["feature_indices_vq"] = bitstream.getvalue()
    compressed_data.pop("feature_indices_vq") # we will save this separately as a binary file, since it is large and has a simple distribution that compresses well with AC
    
    bitout.close()

def compress_vq_gaussians(compressed_data, fname):
    
    data_vqi_gaussian = compressed_data["gaussian_indices_vq"]
    
    # bitstream = BytesIO()
    bitstream = open(fname, "wb")
    
    bitout = BitOutputStream(bitstream)
    default_freq = FlatFrequencyTable(2**12+1)
    freq_models = SimpleFrequencyTable(default_freq)
    
    encoder = ArithmeticEncoder(32, bitout)
    
    for value in data_vqi_gaussian:
        
        encoder.write(freq_models, value)
        freq_models.increment(value)
    
    encoder.write(freq_models, 2**12)  # EOF symbol
    encoder.finish()
   
    bitout.end()
        
    #compressed_data["gaussian_indices_vq"] = bitstream.getvalue()
    compressed_data.pop("gaussian_indices_vq") # we will save this separately as a binary file, since it is large and has a simple distribution that compresses well with AC
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
    compress_vq_features(compressed_data, "feature_indices_vq.bin")
    
    
    ##################################################
    ### VQ indices - gaussian_indices_vq
    ##################################################

    compress_vq_gaussians(compressed_data, "gaussian_indices_vq.bin")


    np.savez_compressed("compressed_ac.npz", **compressed_data)
    
    
    # https://www.nayuki.io/page/reference-arithmetic-coding
    
def decode_features_rest_ac(fname):
    
    feature_shape = (15, 3) # we know the feature shape
    
    bitstream = open(fname, "rb")
    bitin = BitInputStream(bitstream)
    decoder = ArithmeticDecoder(32, bitin)
    
    freq_models = initialize_feature_rest_freq_models(feature_shape)
    
    feature_rest = []
    nb_gaussians = 0
    have_seen_eof = False
    
    while not have_seen_eof:
        nb_gaussians += 1
        sh = np.zeros(feature_shape, dtype=np.uint8)
        
        for sh_par in range(feature_shape[0]):
            for rgb_par in range(feature_shape[1]):
                
                idx = sh_par * feature_shape[1] + rgb_par
                symbol = decoder.read(freq_models[idx])
                
                # check EOF
                if symbol == 256:  # EOF symbol
                    have_seen_eof = True
                    break
                
                freq_models[idx].increment(symbol)
                sh[sh_par, rgb_par] = symbol
            
            if have_seen_eof:
                break
            
        feature_rest.append(sh)
        print(f"Decoded gaussian {nb_gaussians}...", end="\r")
        
        
    feature_rest = (np.stack(feature_rest, dtype=np.int16) - 128).astype(np.int8) # shift back to signed
    
    # _raw_data = np.load('raw_decomp_unc.npz', allow_pickle=True)
    # _raw_data['features_rest'][0, ...]
    
    bitin.close()

def decoder_gs_params():
    
    decode_features_rest_ac("features_rest.bin")
    
    
  
if __name__ == "__main__":
    
    # example encoding features_rest, gaussian_indices_vq and feature_indices_vq with arithmetic coding
    encoder_gs_params()
    
    # exmple decoding features_rest
    decoder_gs_params()