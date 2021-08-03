#!/bin/bash
sudo docker run --rm -d -p 3333:3333 inemo/isanlp
sudo docker run --rm -d -p 3334:3333 inemo/isanlp_deep_srl
