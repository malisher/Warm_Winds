docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p127.0.0.1:1010:22 \
            --gpus '"device=0,1"'\
            --name sku_rec \
            test_sku:latest

