#!/bin/sh

mkdir -p /repo/model_repository_trt/superpoint_trt/1
output_file=/repo/model_repository_trt/superpoint_trt/1/superpoint_fp16.plan
if [ -f ${output_file} ]; then
  echo "============== SuperPoint plan file exist =============="
else
  echo "============== Create SuperPoint plan file =============="
	/usr/src/tensorrt/bin/trtexec --onnx=/repo/model_repository/superpoint_onnx/1/superpoint_fp16.onnx \
			--minShapes='image':1x1x120x240,'keypoint_threshold':1x1 \
			--optShapes='image':1x1x540x960,'keypoint_threshold':1x1 \
			--maxShapes='image':1x1x1080x1920,'keypoint_threshold':1x1 \
		  --precisionConstraints="obey" \
		  --fp16 \
		  --layerPrecisions="GridSample_210:fp32" \
		  --saveEngine=${output_file}
fi

# output_file=/repo/model_repository_trt/superpoint_large_trt/1/superpoint_fp16.plan
# if [ -f ${output_file} ]; then
#     echo "============== SuperPoint large plan file exist =============="
# else
# 	./trtexec --onnx=/repo/model_repository_trt/superpoint_large_trt/superpoint_opt_fp32.onnx \
# 			--minShapes='image':1x120x240x1,'keypoint_threshold':1x1 \
# 			--optShapes='image':1x1080x1920x1,'keypoint_threshold':1x1 \
# 			--maxShapes='image':1x1080x1920x1,'keypoint_threshold':1x1 \
# 			--device=0 \
# 		  --workspace=22000 \
# 		  --precisionConstraints="obey" \
# 			--plugins=/workspace/libdeepmirror_plugin.so \
# 		  --fp16 \
# 		  --layerPrecisions="GridSample_210:fp32" \
# 		  --saveEngine=${output_file}
# fi

mkdir -p /repo/model_repository_trt/lightglue_trt/1
output_file=/repo/model_repository_trt/lightglue_trt/1/lightglue_fp16.plan
if [ -f ${output_file} ]; then
  echo "============== LightGlue plan file exist =============="
else
  echo "============== Create LightGlue plan file =============="
	/usr/src/tensorrt/bin/trtexec --onnx=/repo/model_repository/lightglue_onnx/1/lightglue_fp16.onnx \
		  --fp16 \
		  --precisionConstraints=obey \
		  --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
		  --saveEngine=${output_file}
fi

# output_file=/repo/model_repository_trt/lightglue_large_trt/1/lightglue_v2.plan
# if [ -f ${output_file} ]; then
#     echo "============== LightGlue large plan file exist =============="
# else
# 	./trtexec --onnx=/repo/model_repository_trt/lightglue_large_trt/lightglue_v2.bak.onnx \
# 			--minShapes='kpts0':1x1x2,'kpts1':1x1x2,'desc0':1x1x256,'desc1':1x1x256,'img_shape0':1x2,'img_shape1':1x2,'match_threshold':1x1 \
# 			--optShapes='kpts0':1x1535x2,'kpts1':1x1546x2,'desc0':1x1535x256,'desc1':1x1546x256,'img_shape0':1x2,'img_shape1':1x2,'match_threshold':1x1 \
# 			--maxShapes='kpts0':1x4096x2,'kpts1':1x4096x2,'desc0':1x4096x256,'desc1':1x4096x256,'img_shape0':1x2,'img_shape1':1x2,'match_threshold':1x1 \
# 		  --workspace=10000 \
# 		  --fp16 \
# 		  --precisionConstraints=obey \
# 		  --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
# 		  --saveEngine=${output_file}
# fi

mkdir -p /repo/model_repository_trt/depthanything3_trt/1
output_file=/repo/model_repository_trt/depthanything3_trt/1/da3_small_2_504x280.plan
if [ -f ${output_file} ]; then
  echo "============== DA3 plan file exist =============="
else
  echo "============== Create DA3 plan file =============="
  onnx_model=/repo/model_repository/depthanything3_onnx/1/da3_small_2_504x280.onnx
  # /usr/src/tensorrt/bin/trtexec --onnx=${onnx_model} --dumpLayerInfo --profilingVerbosity=detailed
	/usr/src/tensorrt/bin/trtexec --onnx=${onnx_model} \
		  --fp16 \
		  --precisionConstraints=obey \
		  --saveEngine=${output_file}
fi

output_file=/repo/model_repository_trt/depthanything3_trt/1/da3_small_5_504x280.plan
if [ -f ${output_file} ]; then
  echo "============== DA3 plan file exist =============="
else
  echo "============== Create DA3 plan file =============="
  onnx_model=/repo/model_repository/depthanything3_onnx/1/da3_small_5_504x280.onnx
  # /usr/src/tensorrt/bin/trtexec --onnx=${onnx_model} --dumpLayerInfo --profilingVerbosity=detailed
	/usr/src/tensorrt/bin/trtexec --onnx=${onnx_model} \
		  --fp16 \
		  --precisionConstraints=obey \
		  --saveEngine=${output_file}
fi
