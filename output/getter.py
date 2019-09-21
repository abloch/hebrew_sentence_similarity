#!/bin/bash
aws s3 sync s3://hadashot/output . $*
bunzip2 *.bz2
