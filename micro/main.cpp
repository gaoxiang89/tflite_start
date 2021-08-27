#include "model_r6.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
namespace
{
	tflite::ErrorReporter *error_reporter = nullptr;
	const tflite::Model *model = nullptr;
	tflite::MicroInterpreter *interpreter = nullptr;
	TfLiteTensor *model_input = nullptr;
	TfLiteTensor *model_output = nullptr;

	// Create an area of memory to use for input, output, and intermediate arrays.
	// Finding the minimum value for your model may require some trial and error.
	constexpr uint32_t kTensorArenaSize = 16 * 1024;
	uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring
// additional libraries.
extern const float INPUT_RANGE = 2.f * 3.14159265359f;
// NOTE: extern is used because lcd.c also uses this variable.

// This constant determines the number of inferences to perform across the range
// of x values defined above. Since each inference takes time, the higher this
// number, the more time it will take to run through the entire range. The value
// of this constant can be tuned so that one full cycle takes a desired amount
// of time. Since different devices take different amounts of time to perform
// inference, this value should be defined per-device.
// A larger number than the default to make the animation smoother
const uint16_t INFERENCE_PER_CYCLE = 70;

/* Private user code ---------------------------------------------------------*/
/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = tflite::GetModel(STARK_MODEL);

	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
		TF_LITE_REPORT_ERROR(error_reporter,
							 "Model provided is schema version %d not equal "
							 "to supported version %d.",
							 model->version(), TFLITE_SCHEMA_VERSION);
		return 0;
	}

	// This pulls in all the operation implementations we need.
	static tflite::ops::micro::AllOpsResolver resolver;

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk)
	{
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return 0;
	}

	// Obtain pointers to the model's input and output tensors.
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

	// We are dividing the whole input range with the number of inference
	// per cycle we want to show to get the unit value. We will then multiply
	// the unit value with the current position of the inference
	float unitValuePerDevision = INPUT_RANGE / static_cast<float>(INFERENCE_PER_CYCLE);

	while (1)
	{
		// Calculate an x value to feed into the model
		for (uint16_t inferenceCount = 0; inferenceCount <= INFERENCE_PER_CYCLE; inferenceCount++)
		{
			float x_val = static_cast<float>(inferenceCount) * unitValuePerDevision;

			// Place our calculated x value in the model's input tensor
			model_input->data.f[0] = x_val;

			// Run inference, and report any error
			TfLiteStatus invoke_status = interpreter->Invoke();
			if (invoke_status != kTfLiteOk)
			{
				TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n", static_cast<float>(x_val));
				return 0;
			}

			// Read the predicted y value from the model's output tensor
			float y_val = model_output->data.f[0];

			// Do something with the results
		}
	}
}
