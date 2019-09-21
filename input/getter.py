#!/bin/bash
aws s3 sync s3://hadashot/input . $*
bunzip2 *.bz2
