/***** Includes *****/
#include <stdio.h>
#include <stdint.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "uart.h"
#include "led.h"
#include "board.h"

#include "camera.h"
#include "utils.h"
#include "dma.h"

#include <stdlib.h>
#include <string.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Stopwatch

/*
If BUTTON is defined, you'll need to push PB1 to capture an image frame.  Otherwise, images
will be captured continuously.
*/
//#define BUTTON

#define CAMERA_FREQ (8330000)


#define IMAGE_XRES 88
#define IMAGE_YRES 88

#define CON_BAUD 115200 * 1

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 3-channel 88x88 data input (23232 bytes total / 7744 bytes per channel):
// HWC 88x88, channels 0 to 2
#define INPUT_SIZE 7744
static uint32_t input_0[INPUT_SIZE];

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, INPUT_SIZE);
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}


void process_img(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);

    utils_send_img_to_pc(raw, imgLen, w, h, camera_get_pixel_format());
    
    for (int i = 0; i < h * w * 2; i+=2)
    {
        // Extract colors from RGB565 and convert to signed value
        uint8_t ur = (raw[i] & 0xF8) ^ 0x80;
        uint8_t ug = ((raw[i] << 5) | ((raw[i + 1] & 0xE0) >> 3)) ^ 0x80;
        uint8_t ub = (raw[i + 1] << 3) ^ 0x80;
        input_0[i/2] = 0x00FFFFFF & ((ub << 16) | (ug << 8) | ur);
    }

    load_input(); // Load data input
    cnn_start(); // Start CNN processing

    while (cnn_time == 0)
        MXC_LP_EnterSleepMode(); // Wait for CNN

    softmax_layer();

    printf("Classification results:\n");
    int digs, tens;
    int max_digs = 0;
    int max_tens = 0;
    int class = -1;
    for (int i = 0; i < CNN_NUM_OUTPUTS; i++) {
        digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
        tens = digs % 10;
        digs = digs / 10;
        if (digs > max_digs || (digs == max_digs && tens > max_tens)) {
            max_digs = digs;
            max_tens = tens;
            class = i;
        }
        printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
    }
    printf("Detected : %d\n", class);
    printf("Approximate inference time: %u us\n", cnn_time);
}

// *****************************************************************************
int main(void)
{
    int ret = 0;
    int slaveAddress;
    int id;
    int dma_channel;

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Set system clock to 100 MHz */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    // Initialize DMA for camera interface
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    mxc_uart_regs_t *ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);

    if ((ret = MXC_UART_Init(ConsoleUart, CON_BAUD, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
        return ret;
    }

    // Initialize the camera driver.
    camera_init(CAMERA_FREQ);
    printf("\n\nCamera Example\n");

    slaveAddress = camera_get_slave_address();
    printf("Camera I2C slave address: %02x\n", slaveAddress);

    // Obtain the manufacturer ID of the camera.
    ret = camera_get_manufacture_id(&id);

    if (ret != STATUS_OK) {
        printf("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }

    printf("Camera ID detected: %04x\n", id);


    ret = camera_setup(IMAGE_XRES, IMAGE_YRES, PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA,
                       dma_channel); // RGB565



    if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }

    MXC_Delay(SEC(2));

    // Initialize the CNN, clock at 50 MHz
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    printf("\n*** CNN CONFIGURATION ***\n");

    cnn_init(); // Bring state machine into consistent state
    cnn_load_weights(); // Load kernels
    cnn_load_bias();
    cnn_configure(); // Configure state machine

    printf("\n*** CONFIGURATION DONE ***\n\n");

    // Start capturing a first camera image frame.
    printf("Starting\n");
#ifdef BUTTON
    while (!PB_Get(0)) {}
#endif

    camera_start_capture_image();

    while (1) {
        // Check if image is acquired.
        if (camera_is_image_rcv())
        {
            // Process the image, send it through the UART console.
            process_img();

            // Prepare for another frame capture.
            LED_Toggle(LED_GREEN);
#ifdef BUTTON
            while (!PB_Get(0)) {}
#endif

            MXC_Delay(MSEC(100));
            camera_start_capture_image();
        }
    }

    return ret;
}

/*
  SUMMARY OF OPS
  Hardware: 2,857,344 ops (2,741,056 macc; 116,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 867,328 ops (836,352 macc; 30,976 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 604,032 ops (557,568 macc; 46,464 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 580,800 ops (557,568 macc; 23,232 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 569,184 ops (557,568 macc; 11,616 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 234,400 ops (230,400 macc; 4,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 1,600 ops (1,600 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 16,972 bytes out of 442,368 bytes total (3.8%)
  Bias memory:   2 bytes out of 2,048 bytes total (0.1%)
*/
