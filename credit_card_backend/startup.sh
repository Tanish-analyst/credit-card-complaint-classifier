#!/bin/bash
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 1 main:app
