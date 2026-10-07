# Traffic Surveillance System

A modular traffic surveillance system for vehicle detection, tracking,
categorization, and directional counting.

## Features

- Vehicle detection using Ultralytics YOLO
- Multi-object tracking using ByteTrack
- Active-side filtering using a polygon ROI
- Directional IN and OUT counting using two lines
- Vehicle categorization into 2 Wheeler, 4 Wheeler, and Heavy Vehicle
- Annotated output video
- JSON and CSV count reports
- Structured application logging
- Graceful shutdown support

## Project Structure

    traffic_surveillance/
    ├── config/
    │   └── global_config.py
    ├── core/
    │   ├── detector.py
    │   ├── model_loader.py
    │   └── tracker.py
    ├── logic/
    │   ├── line_counter.py
    │   ├── roi_filter.py
    │   └── vehicle_mapper.py
    ├── output/
    │   ├── stats_generator.py
    │   └── video_writer.py
    ├── pipeline/
    │   ├── frame_capture.py
    │   └── frame_processor.py
    ├── utils/
    │   ├── kill_switch.py
    │   └── logger.py
    ├── main.py
    └── requirements.txt

## Installation

Install the dependencies:

    python -m pip install -r requirements.txt

## Configuration

Edit `config/global_config.py` to configure:

- Input and output video paths
- ROI polygon
- Entry line
- Exit line
- Confidence and IoU thresholds
- YOLO model
- Tracker type

The default Google Colab input path is:

    /content/input.mp4

## Run

    python main.py

## Outputs

The application generates:

- An annotated traffic video
- Vehicle counts in JSON format
- Vehicle counts in CSV format
- A structured log file

## Count Format

    {
      "2_wheeler": {
        "in": 0,
        "out": 0
      },
      "4_wheeler": {
        "in": 0,
        "out": 0
      },
      "heavy_vehicle": {
        "in": 0,
        "out": 0
      }
    }

## Scope

This project performs only:

- Vehicle detection
- Vehicle tracking
- Directional vehicle counting
- Active-side road filtering

It does not perform speed detection, violation detection, face recognition,
or license-plate recognition.
