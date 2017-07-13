#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  //Lequan add 
   // read the weight for each class
  if (this->layer_param_.loss_param().has_weight_source()) {
    const string& weight_source = this->layer_param_.loss_param().weight_source();
    LOG(INFO) << "Opening file " << weight_source;
    std::fstream infile(weight_source.c_str(), std::fstream::in);
    CHECK(infile.is_open());

    Dtype tmp_val;
    while (infile >> tmp_val) {
      CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
      loss_weights_.push_back(tmp_val);
    }
    infile.close();    

    CHECK_EQ(loss_weights_.size(), prob_.shape(1));
  } else {
    LOG(INFO) << "Weight_Loss file is not provided. Assign all one to it.";
    loss_weights_.assign(prob_.shape(1), 1.0);
  }
  //end Lequan
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  //Lequan add 
  vector<int> loss_weights_dim(1);
  loss_weights_dim[0] = loss_weights_.size();
  loss_weights_blob.Reshape(loss_weights_dim);
  Dtype* loss_weights_data = loss_weights_blob.mutable_cpu_data();
  std::copy(loss_weights_.begin(), loss_weights_.end(), loss_weights_data);
}

template <typename Dtype>
Dtype SoftmaxWithWeightedLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  // Lequan add
  //int count = 0;
  Dtype batch_weight = 0;
  //end Lequan
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));

      // Lequan modify
      loss -= loss_weights_[label_value] * log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      //++count;
      batch_weight += loss_weights_[label_value];
      //end Lequan

    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, batch_weight);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    //Lequan modeify
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    // end Lequan
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    
    // Lequan add
    //int count = 0;
    Dtype batch_weight = 0;
    //end Lequan
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          //Lequan modeify
          // for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          //   bottom_diff[i * dim + c * inner_num_ + j] = 0;
          // }
          //
        } else {
          //Lequan add 
          // bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          // ++count;
          batch_weight += loss_weights_[label_value];
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 
              loss_weights_[label_value] * prob_data[i * dim + c * inner_num_ + j];
          }
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 
            loss_weights_[label_value];
          //end Lequan
        }
      }
    }
    // Scale gradient
    //Lequan modify
    //Dtype loss_weight = top[0]->cpu_diff()[0] /
    //                    get_normalizer(normalization_, count);
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, batch_weight);
    //end Lequan
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithWeightedLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithWeightedLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithWeightedLoss);

}  // namespace caffe