TARGET := tflite_micro_test
OUTPUT_DIR := _build
CXX = xt-clang++
SRC_DIR := .
TFLITE_SRC_DIR := tensorflow

SRC_FILES += \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/debug_log.c \
	$(TFLITE_SRC_DIR)/tensorflow/lite/c/common.c \
	$(TFLITE_SRC_DIR)/tensorflow/lite/core/api/error_reporter.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/core/api/op_resolver.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/core/api/tensor_utils.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/kernels/kernel_util.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/kernels/internal/quantization_util.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/activations.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/add.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/all_ops_resolver.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/arg_min_max.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/ceil.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/circular_buffer.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/comparisons.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/concatenation.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/conv.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/depthwise_conv.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/dequantize.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/elementwise.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/floor.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/fully_connected.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/l2norm.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/logical.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/logistic.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/maximum_minimum.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/mul.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/neg.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/pack.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/pad.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/pooling.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/prelu.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/quantize.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/reduce.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/reshape.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/round.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/softmax.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/split.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/strided_slice.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/sub.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/svdf.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/kernels/unpack.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/memory_helpers.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_allocator.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_error_reporter.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_optional_debug_tools.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_string.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_time.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/micro_utils.cc \
	$(TFLITE_SRC_DIR)/tensorflow/lite/micro/simple_memory_allocator.cc \


default:
	$(CXX) -lstdc++  -std=c++11 -fpermissive   $(SRC_FILES) -I$(TFLITE_SRC_DIR)  -Itenserflow -Ithird_party/flatbuffers/include -Ithird_party/gemmlowp -Ithird_party/ruy -o $(TARGET) 
	
.PHONY clean:
	rm $(TARGET)
