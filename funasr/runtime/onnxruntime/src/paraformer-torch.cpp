/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include <sys/time.h>

using namespace std;

namespace funasr {

ParaformerTorch::ParaformerTorch(){
}

void ParaformerTorch::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // knf options
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
    fbank_opts.frame_opts.window_type = "hamming";
    fbank_opts.frame_opts.frame_shift_ms = 10;
    fbank_opts.frame_opts.frame_length_ms = 25;
    fbank_opts.energy_floor = 0;
    fbank_opts.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts);

    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());

    torch::DeviceType device = at::kCPU;
    #ifdef USE_GPU
    if (!torch::cuda::is_available()) {
        LOG(ERROR) << "CUDA is not available! Please check your GPU settings";
        exit(-1);
    } else {
        LOG(INFO) << "CUDA available! Running on GPU";
        device = at::kCUDA;
    }
    #endif
    #ifdef USE_IPEX
    torch::jit::setTensorExprFuserEnabled(false);
    #endif
    torch::jit::script::Module model = torch::jit::load(am_model, device);
    model_ = std::make_shared<TorchModule>(std::move(model));
}

ParaformerTorch::~ParaformerTorch()
{
    if(vocab)
        delete vocab;
    printf("DataZeros takes %f \n", (double)data_zeros_times / 1000);
    printf("Infer takes %f \n", (double)infer_times / 1000);
}

void ParaformerTorch::Reset()
{
}

vector<float> ParaformerTorch::FbankKaldi(float sample_rate, const float* waves, int len) {
    knf::OnlineFbank fbank_(fbank_opts);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());
    //fbank_->InputFinished();
    int32_t frames = fbank_.NumFramesReady();
    int32_t feature_dim = fbank_opts.mel_opts.num_bins;
    vector<float> features(frames * feature_dim);
    float *p = features.data();

    for (int32_t i = 0; i != frames; ++i) {
        const float *f = fbank_.GetFrame(i);
        std::copy(f, f + feature_dim, p);
        p += feature_dim;
    }

    return features;
}

void ParaformerTorch::LoadCmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filename;
        exit(0);
    }
    string line;

    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (line_item[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream means_lines_stream(line);
            vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
            if (means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1; j++) {
                    means_list.push_back(stof(means_lines[j]));
                }
                continue;
            }
        }
        else if (line_item[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vars_lines_stream(line);
            vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
            if (vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1; j++) {
                    vars_list.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

string ParaformerTorch::GreedySearch(float * in, int n_len,  int64_t token_nums)
{
    vector<int> hyps;
    int Tmax = n_len;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        FindMax(in + i * token_nums, token_nums, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->Vector2StringV2(hyps);
}

vector<float> ParaformerTorch::ApplyLfr(const std::vector<float> &in) 
{
    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void ParaformerTorch::ApplyCmvn(std::vector<float> *v)
  {
    int32_t dim = means_list.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + means_list[k]) * vars_list[k];
      }

      p += dim;
    }
  }

vector<std::string> ParaformerTorch::Forward(float** din, int* len, int* flag, int batch)
{
    struct timeval start, end;
    counts += 1;
    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    int32_t feature_dim = lfr_window_size*in_feat_dim;

    std::vector<vector<float>> feats_batch;
    std::vector<int32_t> paraformer_length;
    int max_size = 0;
    int max_frames = 0;
    for(int index=0; index<batch; index++){
        std::vector<float> wav_feats = FbankKaldi(MODEL_SAMPLE_RATE, din[index], len[index]);
        wav_feats = ApplyLfr(wav_feats);
        ApplyCmvn(&wav_feats);
        feats_batch.emplace_back(wav_feats);
        int32_t num_frames  = wav_feats.size() / feature_dim;
        paraformer_length.emplace_back(num_frames);
        if(max_size < wav_feats.size()){
            max_size = wav_feats.size();
            max_frames = num_frames;
        }
    }

    torch::NoGradGuard no_grad;
    // padding
    std::vector<float> all_feats(batch * max_frames * feature_dim);
    for(int index=0; index<batch; index++){
        feats_batch[index].resize(max_size);
        std::memcpy(&all_feats[index * max_frames * feature_dim], feats_batch[index].data(),
                        max_frames * feature_dim * sizeof(float));
    }
    torch::Tensor feats =
        torch::from_blob(all_feats.data(),
                {batch, max_frames, feature_dim}, torch::kFloat).contiguous();
    torch::Tensor feat_lens = torch::from_blob(paraformer_length.data(),
                        {batch}, torch::kInt32);

    // 2. forward
    #ifdef USE_GPU
    feats = feats.to(at::kCUDA);
    feat_lens = feat_lens.to(at::kCUDA);
    #endif
    std::vector<torch::jit::IValue> inputs = {feats, feat_lens};

    vector<std::string> results;
    try {
        gettimeofday(&start, NULL);
        auto outputs = model_->forward(inputs).toTuple()->elements();
        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long infer_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        if(counts > 10){
            infer_times += infer_micros;
        }
        printf("Infer takes %f \n", (double)infer_micros / 1000);
    
        torch::Tensor am_scores;
        torch::Tensor valid_token_lens;
        #ifdef USE_GPU
        am_scores = outputs[0].toTensor().to(at::kCPU);
        valid_token_lens = outputs[1].toTensor().to(at::kCPU);
        #else
        am_scores = outputs[0].toTensor();
        valid_token_lens = outputs[1].toTensor();
        #endif

        for(int index=0; index<batch; index++){
            std::string result = GreedySearch(am_scores[index].data_ptr<float>(), valid_token_lens[index].item<int>(), am_scores.size(2));
            results.emplace_back(result);
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    return results;
}

string ParaformerTorch::ForwardChunk(float* din, int len, int flag)
{

    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}

string ParaformerTorch::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
