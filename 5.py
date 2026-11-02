import os
import gc
import numpy
import numpy as np
import torch
import torchcodec
import datetime
import sounddevice as sd
import soundfile as sf
import multiprocessing as mp
import time
import struct
import usb.core
import usb.util
import libusb_package
from typing import Union
from collections import Counter
from diffusers import StableDiffusionPipeline
from toolbox.toolbox import get_time_str, scale_minmax, DirectoryAccessor


class ReSpeaker:
    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev

    def read(self,):
        try:
            data = (20, 18, 4, "ro", "uint16")
        except KeyError:
            return

        resid = data[0]
        cmdid = 0x80 | data[1]
        length = data[2] + 1  # 1 byte for status

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, length, self.TIMEOUT)

        result = response.tolist()

        return result[3] == 1

    def close(self):
        """
        close the interface
        """
        usb.util.dispose_resources(self.dev)


class AudioPrep:
    import io
    import torch
    import torchaudio
    from torchcodec.decoders import AudioDecoder
    from torchvision.transforms import v2
    from scipy.io import wavfile

    def __init__(self,):
        self.sample_rate = 48000
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128

    def _scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def process_mp3(self, audio: np.ndarray):
        """
        Worker function to apply workers to files.
        """
        try:
            audio = self.torch.from_numpy(audio)
            bytes_io = self.io.BytesIO()
            self.wavfile.write(bytes_io, self.sample_rate, audio)
            dec = self.AudioDecoder(bytes_io.getvalue(), num_channels=1)
            y = dec.get_all_samples()
            y = y.data

            # converting audio np array to spectrogram
            mel_spectrogram_transform = self.torchaudio.transforms.MelSpectrogram(
                sample_rate=dec.metadata.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            mel_spec = mel_spectrogram_transform(y.data)
            to_db = self.torchaudio.transforms.AmplitudeToDB(stype="power")
            mel_spec_db = to_db(mel_spec)
            mel_spec_db = mel_spec_db.detach()
            resize_tool = self.v2.Resize((64, 256))
            img = resize_tool(mel_spec_db)
            img = scale_minmax(img,)

            return img

        except Exception as e:
            return False, (f"Error processing: {e}",)


# class BirdID:
#     import tensorrt as trt
#     # import tensorrt_rtx as trt
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#
#     def __init__(self, engine_path: str="BirdID_US.engine",):
#         # 1. Initialize Logger and Runtime
#         self.logger = self.trt.Logger(self.trt.Logger.WARNING)
#         self.runtime = self.trt.Runtime(self.logger)
#
#         # 2. Load the Serialized Engine
#         self.engine_path = engine_path
#         self.engine = self._load_engine(self.engine_path)
#
#
#         # 3. Create Execution Context and Allocate Device Memory
#         self.context = self.engine.create_execution_context()
#         self.stream = self.cuda.Stream()
#
#         # self.inputs = []
#         # self.outputs = []
#         # self.bindings = []
#
#         self.data = np.array(0, dtype=np.float32)
#         self.d_input = np.array(0, dtype=np.float32)
#         self.d_output = np.array(0, dtype=np.float32)
#
#     def _load_engine(self, engine_path: str):
#         with open(engine_path, "rb") as f:
#             self.engine = self.runtime.deserialize_cuda_engine(f.read())
#         return self.engine
#
#     def prepare(self, data: Union[np.ndarray, torch.Tensor]):
#         if data is not None:
#             if isinstance(data, torch.Tensor):
#                 data = data.numpy()
#             input_shape = data.shape
#         else:
#             raise TypeError
#         output_shape = (data.shape[0], 1729,)
#         tensor_name = self.engine.get_tensor_name(0)
#         self.context.set_input_shape(tensor_name, input_shape)
#         assert self.context.all_binding_shapes_specified
#
#         self.d_input = self.cuda.mem_alloc(np.prod(data.shape) * np.dtype(data.dtype).itemsize)
#         self.d_output = self.cuda.mem_alloc(np.prod(output_shape) * np.dtype(data.dtype).itemsize)
#         # input buffer
#         self.context.set_tensor_address(self.engine.get_tensor_name(0), int(self.d_input))
#         # output buffer
#         self.context.set_tensor_address(self.engine.get_tensor_name(1), int(self.d_output))
#
#         # for binding in self.engine:
#         #     size = self.trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
#         #     dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))
#         #
#         #     # Allocate host and device buffers
#         #     host_mem = self.cuda.pagelocked_empty(size, dtype)
#         #     device_mem = self.cuda.alloc(host_mem.nbytes)
#         #
#         #     # Append the device buffer memory pointer to bindings
#         #     self.bindings.append(int(device_mem))
#         #
#         #     # Classify buffers into inputs and outputs
#         #     if self.engine.binding_is_input(binding):
#         #         self.inputs.append({'host': host_mem, 'device': device_mem})
#         #     else:
#         #         self.outputs.append({'host': host_mem, 'device': device_mem})
#
#     def predict(self,):
#         # 4. Prepare Input Data
#         # Ensure data shape matches the model's exact expected shape
#         if isinstance(self.data, torch.Tensor):
#             self.data = self.data.numpy()
#         # np.copyto(self.inputs[0]["host"], self.data.ravel())
#
#         # 5. Execute Inference
#         # Copy input data from Host RAM to Jetson/GPU unified Device RAM
#         self.cuda.memcpy_htod_async(self.d_input, self.data, self.stream)
#
#         # Execute the model asynchronously
#         self.context.execute_async_v3(stream_handle=self.stream.handle)
#
#         # Copy predictions back from Device RAM to Host RAM
#         output_data = np.empty(self.d_output.shape, dtype=np.float32)
#         self.cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
#
#         # Synchronize the stream to wait for completion
#         self.stream.synchronize()
#
#         # 6. Extract Predictions
#         predictions = self.outputs[0]['host']
#         predictions = self.trt.nptype(self.engine.get_tensor_dtype(predictions))
#
#         return predictions


class TensorRTInference:
    import tensorrt as trt
    # import tensorrt_rtx as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    def __init__(self, engine_path: str="BirdID_US.trt",):
        self.logger = self.trt.Logger(self.trt.Logger.ERROR)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = self.cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = self.trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = self.trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data: Union[np.ndarray, torch.Tensor], return_logits: bool=False,):
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()

        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        self.cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()
        # self.stream.synchronize(self.cuda.Stream)

        return self.outputs[0].host if return_logits else np.argmax(self.outputs[0].host)


class BirdArtist:
    from pathlib import Path
    def __init__(self,):
        self.hourly_birds = []

    def __call__(self, name: str,):
        # if not self.hourly_birds:
        #     print("No birds detected this hour. Skipping image generation.")
        #     return
        # # Determine 5 least common birds
        # counts = Counter(self.hourly_birds)
        # # .most_common() returns descending, so [::-1] makes it ascending (least common first)
        # least_common_ordered = [bird for bird, count in counts.most_common()[::-1]]

        # final_five = []
        # idx = 0
        # # If less than 5 unique birds exist, repeat them in order to reach 5
        # while len(final_five) < 5:
        #     final_five.append(least_common_ordered[idx % len(least_common_ordered)])
        #     idx += 1

        # bird_names_str = ", ".join(final_five)
        # prompt = (f"A 19th-century vintage ornithology field guide book illustration of "
        #           f"the following birds: {bird_names_str}. Detailed, scientific illustration style, "
        #           f"aged antique paper, sketched, watercolor notes.")
        prompt = (f"A 19th-century vintage ornithology field guide book illustration of "
                  f"the bird: {name}. Detailed, scientific illustration style, "
                  f"aged antique paper, sketched, watercolor notes.")

        # Dynamically load SD to avoid colliding with TensorRT memory footprint
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        # Generate and save image
        image = pipe(prompt, num_inference_steps=25).images[0]
        img_dir = "images/"
        self.Path(img_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{img_dir}field_guide_{get_time_str()}.png"
        image.save(filename)
        print(f"Hourly image saved to {filename}")

        # Unload model and free memory
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

        # # Reset hourly log
        # self.hourly_birds = []


def find(vid=0x2886, pid=0x001A):
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if not dev:
        return None

    return ReSpeaker(dev)


def activate(e: mp.synchronize.Event, ee: mp.synchronize.Event, qq: mp.queues.Queue,):
    dev = find()
    try:
        while True:
            while not e.is_set() and not ee.is_set() and qq.empty() and dev.read():
                # print(dev.read())
                qq.put(True)
                ee.set()
            time.sleep(1)
            if e.is_set():
                break
    except KeyboardInterrupt:
        dev.close()


def record(e: mp.synchronize.Event, ee: mp.synchronize.Event, q: mp.queues.Queue, qq: mp.queues.Queue,):
    while True:
        while not e.is_set() and (not ee.is_set() or not q.empty()):
            flag = q.get()
            if flag:
                mic = -1
                duration = 10
                sample_rate = 48000
                channels = 2
                for idx, device in enumerate(sd.query_devices()):
                    if "reSpeaker XVF3800 4-Mic Array" in device["name"]:
                        mic = idx
                audio = sd.rec(
                    frames=int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=channels,
                    blocking=True,
                    device=mic,
                )
                # # save audio (lossless)
                # from scipy.io.wavfile import write
                # name = f"audio_recording_{get_time_str()}.wav"
                # write(name, sample_rate, audio)

                # audio = torch.from_numpy(audio)
                qq.put(audio)
        time.sleep(1)
        if e.is_set():
            break


def predict(e: mp.synchronize.Event, q: mp.queues.Queue, qq: mp.queues.Queue,):
    def log_prediction(bird_name: str):
        from pathlib import Path
        dir_path = "log/"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        now = get_time_str("%Y%m%d")
        file_path = f"birdid_log_{now}.txt"
        writer = DirectoryAccessor(dir_path=dir_path, file_path=file_path, flags="a")
        writer.write(f"{bird_name},{now}")
        print(f"Logged Prediction: {bird_name}")

    import pandas as pd
    mel = AudioPrep()
    hourly_birds = []
    class_mappings = pd.read_csv("class_mapping_us.csv", index_col="ClassID")
    # model = BirdID()
    model = TensorRTInference()
    while True:
        while not e.is_set() and not q.empty():
            import random
            audio = q.get()
            spec = mel.process_mp3(audio)

            # Get prediction from model
            # model.prepare(spec)
            # pred = model.predict()

            pred = model.infer(spec)  # output shape (batch_size, n_classes,)

            if pred.size == 1:
                # name = class_mappings.loc[np.argmax(pred), "ClassName"]
                name = class_mappings.loc[pred, "ClassName"]
                hourly_birds.append(name)
                log_prediction(name,)
                chosen = name
            else:
                if len(pred.shape) == 2: pred = np.squeeze(pred)
                for i in pred:
                    # name = class_mappings.loc[np.argmax(i), "ClassName"]
                    name = class_mappings.loc[i, "ClassName"]
                    hourly_birds.append(name)
                    log_prediction(name,)
                chosen = random.choice(hourly_birds)
                # [hourly_birds.append(np.argmax(i)) for i in pred]

            qq.put(chosen)
        time.sleep(1)
        if e.is_set():
            break


def draw(e: mp.synchronize.Event, q: mp.queues.Queue,):
    while True:
        while not e.is_set():
            name = q.get()

            print(name)  # Placeholder, need to add code to get model to create image
            artist = BirdArtist()
            artist(name)
        time.sleep(1)
        if e.is_set():
            break


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    e = mp.Event()
    ee = mp.Event()
    flag_queue = mp.Queue(1)
    data_queue = mp.Queue(1)
    predict_queue = mp.Queue(1)
    a = mp.Process(target=activate, args=(e, ee, flag_queue,))
    r = mp.Process(target=record, args=(e, ee, flag_queue, data_queue,))
    p = mp.Process(target=predict, args=(e, data_queue, predict_queue,))
    d = mp.Process(target=draw, args=(e, predict_queue,))
    try:
        d.start()
        p.start()
        r.start()
        a.start()
        while True:
            if ee.is_set():
                time.sleep(600)
                ee.clear()
            time.sleep(1)

    except KeyboardInterrupt:
        e.set()
        a.terminate()
        a.close()
        r.terminate()
        r.close()
        p.terminate()
        p.close()
        d.terminate()
        d.close()