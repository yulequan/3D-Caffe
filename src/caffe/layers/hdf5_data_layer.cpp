/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;


  // for (int i = 0; i < top_size; ++i) {
  //   hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  //   hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
  //       MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  // }
  // TODO do random segmentation
  // Lequan add 
  const bool has_crop = this->layer_param_.transform_param().has_crop_size_w();
  vector<shared_ptr<Blob<Dtype> > > hdf_input_blobs;
  hdf_input_blobs.resize(top_size);
  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf_input_blobs[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    if(has_crop)
      hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
          MIN_DATA_DIM, MAX_DATA_DIM, hdf_input_blobs[i].get());
    else
      hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
          MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());      
  }
  if (has_crop){
    if(top_size==2)
      HDF5DataTransform(hdf_input_blobs[0].get(),hdf_blobs_[0].get(),hdf_input_blobs[1].get(),hdf_blobs_[1].get());
    if(top_size==3)
      HDF5DataTransform2(hdf_input_blobs[0].get(),hdf_blobs_[0].get(),hdf_input_blobs[1].get(),hdf_blobs_[1].get(),
       hdf_input_blobs[2].get(),hdf_blobs_[2].get());
  }
  // Lequan end add

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // // Refuse transformation parameters since HDF5 is totally generic.
  // CHECK(!this->layer_param_.has_transform_param()) <<
  //     this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      current_row_ = 0;
      if (this->layer_param_.hdf5_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
  }
}

template<typename Dtype>
void HDF5DataLayer<Dtype>::HDF5DataTransform(Blob<Dtype>* input_blob_data, Blob<Dtype>* transformed_blob_data,
                       Blob<Dtype>* input_blob_label, Blob<Dtype>* transformed_blob_label){
  
  std::vector<int> transformed_data_shape = input_blob_data->shape();
  std::vector<int> transformed_label_shape = input_blob_label->shape();
  
  TransformationParameter transform_param = this->layer_param_.transform_param();
  bool has_crop_size  = transform_param.has_crop_size_w();
  const int crop_size_w = transform_param.crop_size_w();
  const int crop_size_h = transform_param.crop_size_h();
  const int crop_size_l = transform_param.crop_size_l();
  if (has_crop_size){
      transformed_data_shape[2] = crop_size_l;
      transformed_data_shape[3] = crop_size_h;
      transformed_data_shape[4] = crop_size_w;
      transformed_label_shape[2] = crop_size_l;
      transformed_label_shape[3] = crop_size_h;
      transformed_label_shape[4] = crop_size_w;
  }
  

  if (transformed_blob_data->count() == 0) {
    // Initialize transformed_blob_data with the right shape.
    if (has_crop_size) {
      transformed_blob_data->Reshape(transformed_data_shape);
      transformed_blob_label->Reshape(transformed_label_shape);
    } else {
      transformed_blob_data->ReshapeLike(*input_blob_data);
      transformed_blob_label->ReshapeLike(*input_blob_label);
    }
  }
  
  const int input_num = input_blob_data->shape(0);
  const int input_channels = input_blob_data->shape(1);
  const int input_length = input_blob_data->shape(2);
  const int input_height = input_blob_data->shape(3);
  const int input_width = input_blob_data->shape(4);

  const int num = transformed_blob_data->shape(0);
  const int channels = transformed_blob_data->shape(1);
  const int length  = transformed_blob_data->shape(2);
  const int height = transformed_blob_data->shape(3);
  const int width = transformed_blob_data->shape(4);
  const int size = transformed_blob_data->count();

  CHECK_LE(transformed_data_shape[0], num);
  CHECK_EQ(transformed_data_shape[1], channels);
  CHECK_GE(transformed_data_shape[2], length);
  CHECK_GE(transformed_data_shape[3], height);
  CHECK_GE(transformed_data_shape[4], width);


  const Dtype scale = transform_param.scale();
  const bool do_mirror = transform_param.mirror() && Rand(2);
  const bool has_mean_value = transform_param.mean_value_size() > 0;

  int h_off = 0;
  int w_off = 0;
  int l_off = 0;
  if (has_crop_size) {
    CHECK_EQ(crop_size_l, length);
    CHECK_EQ(crop_size_h, height);
    CHECK_EQ(crop_size_w, width);
    // We only do random crop when we do training.
    if (this->phase_ == TRAIN) {
      l_off = Rand(input_length - crop_size_l + 1);
      h_off = Rand(input_height - crop_size_h + 1);
      w_off = Rand(input_width  - crop_size_w + 1);
    } else {
      l_off = (input_length - crop_size_l) /2;
      h_off = (input_height - crop_size_h) / 2;
      w_off = (input_width  - crop_size_w) / 2;
    }
  } else {
    CHECK_EQ(input_length, length);
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }
  
  // transform data
  Dtype* input_data = input_blob_data->mutable_cpu_data();
  if (has_mean_value) {
   // LOG(INFO) << "has mean value.";
    CHECK(transform_param.mean_value_size() == 1 || transform_param.mean_value_size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (transform_param.mean_value_size() == 1) {
      caffe_add_scalar(input_blob_data->count(), -((Dtype)transform_param.mean_value(0)), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = (n*input_channels+c)*input_length*input_height*input_width;
          caffe_add_scalar(input_length*input_height * input_width, -((Dtype)transform_param.mean_value(c)),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob_data->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * length;
      int data_index_c = (data_index_n + c) * input_length + l_off;
      for (int l = 0; l < length; l++){
        int top_index_l = (top_index_c + l) * height;
        int data_index_l = (data_index_c + l) * input_height + h_off;
        for (int h = 0; h < height; ++h) {
          int top_index_h = (top_index_l + h) * width;
          int data_index_h = (data_index_l + h) * input_width + w_off;
          if (do_mirror) {
            int top_index_w = top_index_h + width - 1;
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_w-w] = input_data[data_index_h + w];
            }
          } else {
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_h + w] = input_data[data_index_h + w];
            }
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }

  // transform label
  const int label_channels = transformed_blob_label->shape(1);
  input_data = input_blob_label->mutable_cpu_data();
  transformed_data = transformed_blob_label->mutable_cpu_data();
  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * label_channels;
    int data_index_n = n * label_channels;
    for (int c = 0; c < label_channels; ++c) {
      int top_index_c = (top_index_n + c) * length;
      int data_index_c = (data_index_n + c) * input_length + l_off;
      for (int l = 0; l < length; l++){
        int top_index_l = (top_index_c + l) * height;
        int data_index_l = (data_index_c + l) * input_height + h_off;
        for (int h = 0; h < height; ++h) {
          int top_index_h = (top_index_l + h) * width;
          int data_index_h = (data_index_l + h) * input_width + w_off;
          if (do_mirror) {
            int top_index_w = top_index_h + width - 1;
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_w-w] = input_data[data_index_h + w];
            }
          } else {
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_h + w] = input_data[data_index_h + w];
            }
          }
        }
      }
    }
  }
}

template<typename Dtype>
void HDF5DataLayer<Dtype>::HDF5DataTransform2(Blob<Dtype>* input_blob_data, Blob<Dtype>* transformed_blob_data,
                       Blob<Dtype>* input_blob_label1, Blob<Dtype>* transformed_blob_label1,
                       Blob<Dtype>* input_blob_label2, Blob<Dtype>* transformed_blob_label2){
  std::vector<int> transformed_data_shape = input_blob_data->shape();
  std::vector<int> transformed_label1_shape = input_blob_label1->shape();
  std::vector<int> transformed_label2_shape = input_blob_label2->shape();
  
  TransformationParameter transform_param = this->layer_param_.transform_param();
  bool has_crop_size  = transform_param.has_crop_size_w();
  const int crop_size_w = transform_param.crop_size_w();
  const int crop_size_h = transform_param.crop_size_h();
  const int crop_size_l = transform_param.crop_size_l();
  if (has_crop_size){
      transformed_data_shape[2] = crop_size_l;
      transformed_data_shape[3] = crop_size_h;
      transformed_data_shape[4] = crop_size_w;
      transformed_label1_shape[2] = crop_size_l;
      transformed_label1_shape[3] = crop_size_h;
      transformed_label1_shape[4] = crop_size_w;
      transformed_label2_shape[2] = crop_size_l;
      transformed_label2_shape[3] = crop_size_h;
      transformed_label2_shape[4] = crop_size_w;
  }
  if (transformed_blob_data->count() == 0) {
    // Initialize transformed_blob_data with the right shape.
    if (has_crop_size) {
      transformed_blob_data->Reshape(transformed_data_shape);
      transformed_blob_label1->Reshape(transformed_label1_shape);
      transformed_blob_label2->Reshape(transformed_label2_shape);
    } else {
      transformed_blob_data->ReshapeLike(*input_blob_data);
      transformed_blob_label1->ReshapeLike(*input_blob_label1);
      transformed_blob_label2->ReshapeLike(*input_blob_label2);
    }
  }
  const int input_num = input_blob_data->shape(0);
  const int input_channels = input_blob_data->shape(1);
  const int input_length = input_blob_data->shape(2);
  const int input_height = input_blob_data->shape(3);
  const int input_width = input_blob_data->shape(4);

  const int num = transformed_blob_data->shape(0);
  const int channels = transformed_blob_data->shape(1);
  const int length  = transformed_blob_data->shape(2);
  const int height = transformed_blob_data->shape(3);
  const int width = transformed_blob_data->shape(4);
  const int size = transformed_blob_data->count();

  CHECK_LE(transformed_data_shape[0], num);
  CHECK_EQ(transformed_data_shape[1], channels);
  CHECK_GE(transformed_data_shape[2], length);
  CHECK_GE(transformed_data_shape[3], height);
  CHECK_GE(transformed_data_shape[4], width);


  const Dtype scale = transform_param.scale();
  const bool do_mirror = transform_param.mirror() && Rand(2);
  const bool has_mean_value = transform_param.mean_value_size() > 0;
  int h_off = 0;
  int w_off = 0;
  int l_off = 0;
  if (has_crop_size) {
    CHECK_EQ(crop_size_l, length);
    CHECK_EQ(crop_size_h, height);
    CHECK_EQ(crop_size_w, width);
    // We only do random crop when we do training.
    if (this->phase_ == TRAIN) {
      l_off = Rand(input_length - crop_size_l + 1);
      h_off = Rand(input_height - crop_size_h + 1);
      w_off = Rand(input_width  - crop_size_w + 1);
    } else {
      l_off = (input_length - crop_size_l) /2;
      h_off = (input_height - crop_size_h) / 2;
      w_off = (input_width  - crop_size_w) / 2;
    }
  } else {
    CHECK_EQ(input_length, length);
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }
  
  // transform data
  Dtype* input_data = input_blob_data->mutable_cpu_data();
  if (has_mean_value) {
   // LOG(INFO) << "has mean value.";
    CHECK(transform_param.mean_value_size() == 1 || transform_param.mean_value_size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (transform_param.mean_value_size() == 1) {
      caffe_add_scalar(input_blob_data->count(), -((Dtype)transform_param.mean_value(0)), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = (n*input_channels+c)*input_length*input_height*input_width;
          caffe_add_scalar(input_length*input_height * input_width, -((Dtype)transform_param.mean_value(c)),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob_data->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * length;
      int data_index_c = (data_index_n + c) * input_length + l_off;
      for (int l = 0; l < length; l++){
        int top_index_l = (top_index_c + l) * height;
        int data_index_l = (data_index_c + l) * input_height + h_off;
        for (int h = 0; h < height; ++h) {
          int top_index_h = (top_index_l + h) * width;
          int data_index_h = (data_index_l + h) * input_width + w_off;
          if (do_mirror) {
            int top_index_w = top_index_h + width - 1;
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_w-w] = input_data[data_index_h + w];
            }
          } else {
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_h + w] = input_data[data_index_h + w];
            }
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }

  // transform label1
  const int label1_channels = transformed_blob_label1->shape(1);
  input_data = input_blob_label1->mutable_cpu_data();
  transformed_data = transformed_blob_label1->mutable_cpu_data();
  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * label1_channels;
    int data_index_n = n * label1_channels;
    for (int c = 0; c < label1_channels; ++c) {
      int top_index_c = (top_index_n + c) * length;
      int data_index_c = (data_index_n + c) * input_length + l_off;
      for (int l = 0; l < length; l++){
        int top_index_l = (top_index_c + l) * height;
        int data_index_l = (data_index_c + l) * input_height + h_off;
        for (int h = 0; h < height; ++h) {
          int top_index_h = (top_index_l + h) * width;
          int data_index_h = (data_index_l + h) * input_width + w_off;
          if (do_mirror) {
            int top_index_w = top_index_h + width - 1;
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_w-w] = input_data[data_index_h + w];
            }
          } else {
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_h + w] = input_data[data_index_h + w];
            }
          }
        }
      }
    }
  }

  // transform label2
  const int label2_channels = transformed_blob_label2->shape(1);
  input_data = input_blob_label2->mutable_cpu_data();
  transformed_data = transformed_blob_label2->mutable_cpu_data();
  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * label2_channels;
    int data_index_n = n * label2_channels;
    for (int c = 0; c < label2_channels; ++c) {
      int top_index_c = (top_index_n + c) * length;
      int data_index_c = (data_index_n + c) * input_length + l_off;
      for (int l = 0; l < length; l++){
        int top_index_l = (top_index_c + l) * height;
        int data_index_l = (data_index_c + l) * input_height + h_off;
        for (int h = 0; h < height; ++h) {
          int top_index_h = (top_index_l + h) * width;
          int data_index_h = (data_index_l + h) * input_width + w_off;
          if (do_mirror) {
            int top_index_w = top_index_h + width - 1;
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_w-w] = input_data[data_index_h + w];
            }
          } else {
            for (int w = 0; w < width; ++w) {
              transformed_data[top_index_h + w] = input_data[data_index_h + w];
            }
          }
        }
      }
    }
  }
}


template <typename Dtype>
int HDF5DataLayer<Dtype>::Rand(int n) {
  return rand()%n;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
