cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (cudaBindTexture(0, %BUFFER_NAME%_tex, h.cuda.%BUFFER_NAME%_d, channelDesc, %ELEMENT_COUNT% * sizeof(float)) != cudaSuccess)
            utils::exit_with_err("Could bind texture");
#pragma GCC diagnostic pop