MODEL_PATH="PATH_TO_MIMO_VL_MODEL"
MODEL_HOME=$(dirname ${MODEL_PATH})
MODEL_NAME=$(basename ${MODEL_PATH})

EVAL_RESULTS_DIR="./eval_results"

TASK="ai2d_boxed"

# Environment variables for GPT-4o evaluation
export API_TYPE="openai"
export OPENAI_API_URL="YOUR_OPENAI_API_URL"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export GPT_EVAL_NUM_RETRIES=5
export GPT_EVAL_NUM_SECONDS_TO_SLEEP=10
export GPT_EVAL_RAISE_AFTER_MAX_RETRIES=0
export GPT_EVAL_TIMEOUT=30

# Configuration for image and video preprocessing
PATCH_SIZE=28
IMAGE_MIN_TOKENS=0
IMAGE_MAX_TOKENS=4096
VIDEO_MIN_TOKENS=0
VIDEO_MAX_TOKENS=4096
VIDEO_TOTAL_MAX_TOKENS=16384
VIDEO_FPS=2
VIDEO_MIN_FRAMES=0
VIDEO_MAX_FRAMES=256
VIDEO_NFRAMES=0

IMAGE_MIN_PIXELS=$(($IMAGE_MIN_TOKENS * $PATCH_SIZE * $PATCH_SIZE))
IMAGE_MAX_PIXELS=$(($IMAGE_MAX_TOKENS * $PATCH_SIZE * $PATCH_SIZE))
VIDEO_MIN_PIXELS=$(($VIDEO_MIN_TOKENS * $PATCH_SIZE * $PATCH_SIZE))
VIDEO_MAX_PIXELS=$(($VIDEO_MAX_TOKENS * $PATCH_SIZE * $PATCH_SIZE))
VIDEO_TOTAL_MAX_PIXELS=$(($VIDEO_TOTAL_MAX_TOKENS * $PATCH_SIZE * $PATCH_SIZE))

export QWEN_RESIZE_MAX_PIXELS=$IMAGE_MAX_PIXELS

kwargs=""
if [ ${IMAGE_MIN_PIXELS} -gt 0 ]; then
    kwargs="${kwargs},image_min_pixels=${IMAGE_MIN_PIXELS}"
fi
if [ ${IMAGE_MAX_PIXELS} -gt 0 ]; then
    kwargs="${kwargs},image_max_pixels=${IMAGE_MAX_PIXELS}"
fi
if [ ${VIDEO_MIN_PIXELS} -gt 0 ]; then
    kwargs="${kwargs},video_min_pixels=${VIDEO_MIN_PIXELS}"
fi
if [ ${VIDEO_MAX_PIXELS} -gt 0 ]; then
    kwargs="${kwargs},video_max_pixels=${VIDEO_MAX_PIXELS}"
fi
if [ ${VIDEO_TOTAL_MAX_PIXELS} -gt 0 ]; then
    kwargs="${kwargs},video_total_max_pixels=${VIDEO_TOTAL_MAX_PIXELS}"
fi
if [ ${VIDEO_FPS} -gt 0 ]; then
    kwargs="${kwargs},video_fps=${VIDEO_FPS}"
fi
if [ ${VIDEO_MIN_FRAMES} -gt 0 ]; then
    kwargs="${kwargs},video_min_frames=${VIDEO_MIN_FRAMES}"
fi
if [ ${VIDEO_MAX_FRAMES} -gt 0 ]; then
    kwargs="${kwargs},video_max_frames=${VIDEO_MAX_FRAMES}"
fi
if [ ${VIDEO_NFRAMES} -gt 0 ]; then
    kwargs="${kwargs},video_nframes=${VIDEO_NFRAMES}"
fi

echo "Evaluating ${TASK} with kwargs: ${kwargs}"
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 29529 \
    -m lmms_eval \
    --model mivllm \
    --model_args model_version=${MODEL_HOME}/${MODEL_NAME},max_model_len=128000,gpu_memory_utilization=0.6,max_num_seqs=16,max_images=120,max_videos=10,max_audios=0,dtype=bfloat16${kwargs} \
    --tasks ${TASK} \
    --batch_size 8 \
    --log_samples \
    --log_samples_suffix ${MODEL_NAME} \
    --output_path ${EVAL_RESULTS_DIR}/${MODEL_NAME}