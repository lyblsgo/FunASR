/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once
#include <torch/serialize.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "precomp.h"

namespace funasr {

    class ParaformerTorch : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * ParaformerTorch: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts;

        Vocab* vocab;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void LoadCmvn(const char *filename);
        vector<float> ApplyLfr(const vector<float> &in);
        void ApplyCmvn(vector<float> *v);
        string GreedySearch( float* in, int n_len, int64_t token_nums);

        using TorchModule = torch::jit::script::Module;
        std::shared_ptr<TorchModule> model_ = nullptr;
        std::vector<torch::Tensor> encoder_outs_;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

        long data_zeros_times=0;
        long infer_times=0;
        int counts = 0;

    public:
        ParaformerTorch();
        ~ParaformerTorch();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string ForwardChunk(float* din, int len, int flag);
        vector<std::string> Forward(float** din, int* len, int* flag, int batch);
        string Rescoring();
    };

} // namespace funasr
