#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
g++ -std=c++17 -O2 -pipe main.cpp -o main
