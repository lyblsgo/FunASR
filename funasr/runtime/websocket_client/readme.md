
# Build websocket client
required openssl lib

```shell
apt-get install libssl-dev #ubuntu 
# yum install openssl-devel #centos

mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release ..
make
```

# Run websocket client test

```shell
./funasr-wss-client  --server-ip <string>
                    --port <string>
                    --wav-path <string>
                    [--thread-num <int>] 
                    [--is-ssl <int>]  [--]
                    [--version] [-h]

Where:
   --server-ip <string>
     (required)  server-ip

   --port <string>
     (required)  port

   --wav-path <string>
     (required)  the input could be: wav_path, e.g.: asr_example.wav;
     pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)

   --thread-num <int>
     thread-num

   --is-ssl <int>
     is-ssl is 1 means use wss connection, or use ws connection

example:
./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1

