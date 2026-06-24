#!/usr/bin/env python3

import argparse
import signal
import time
from queue import Queue
from threading import Thread, Event

from toolset.data_sources import FileDataSource
from toolset.data_sources.uart_source import UartDataSource
from toolset.pipeline import producer_worker
from toolset.pipeline.session_logger import SessionLogger
from toolset.processing.cs_subevent_data_consumer import dual_stream_consumer
from toolset.gui.cs_viewer import launch_viewer
from toolset.estimators.phase_slope import PhaseSlopeEstimator
from toolset.estimators.ifft import IFFTEstimator
from toolset.estimators.music import MUSICEstimator
from toolset.estimators.weighted_ls import WeightedLSEstimator
from toolset.estimators.ema import EMAEstimator
from toolset.pipeline.dsp_worker import DSPWorker


def main():
    parser = argparse.ArgumentParser(
        description='Process Bluetooth Channel sounding PBR data'
    )

    parser.add_argument(
        '-i', '--initiator',
        required=True,
        help='Path to initiator log file or COM-port (e.g., /dev/ttyACM1)'
    )

    parser.add_argument(
        '-r', '--reflector',
        required=True,
        help='Path to reflector log file or COM-port (e.g., /dev/ttyACM3)'
    )

    parser.add_argument(
        '--uart',
        action='store_true',
        help='If specified, treat initiator and reflector as COM-ports instead of log files'
    )

    parser.add_argument(
        '--log-uart',
        action='store_true',
        help='Write raw UART data to log/<timestamp>/ folder'
    )

    parser.add_argument(
        '--log-session',
        action='store_true',
        help='Save all console output (stdout + stderr) to log/<timestamp>/session.log'
    )

    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Baudrate for COM-ports (default: 115200)'
    )

    parser.add_argument(
        '--rtscts',
        action='store_true',
        help='Enable hardware flow control for COM-ports'
    )

    parser.add_argument(
        '--ml',
        action='store_true',
        help='Enable the Sensing tab for ML-based features'
    )

    parser.add_argument(
        '--ml-handler',
        metavar='SCRIPT',
        default=None,
        help='Path to a Python script invoked during live ML recognition (requires --ml)'
    )

    theme_group = parser.add_mutually_exclusive_group()
    theme_group.add_argument(
        '--dark',
        dest='dark_mode',
        action='store_true',
        default=True,
        help='Use dark theme (default)'
    )
    theme_group.add_argument(
        '--light',
        dest='dark_mode',
        action='store_false',
        help='Use light theme'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.log_uart and not args.uart:
        parser.error("--log-uart can only be used with --uart")
    if args.ml and not args.uart:
        parser.error("--ml requires --uart")
    if args.ml_handler and not args.ml:
        parser.error("--ml-handler requires --ml")

    # ------------------------------------------------------------------ #
    # Session logger — always created; controls what gets written to disk  #
    # ------------------------------------------------------------------ #
    need_session_folder = args.log_uart or args.log_session
    session_logger: SessionLogger | None = None

    if need_session_folder:
        session_logger = SessionLogger(
            log_root='log',
            enable_session_log=args.log_session,
        )
        print(f"Session folder: {session_logger.session_dir}")

    # Create separate queues for each stream
    initiator_queue = Queue(maxsize=100)
    reflector_queue = Queue(maxsize=100)
    stop_event = Event()

    if args.uart:
        print("Mode: Reading from COM-ports")
        initiator_source = UartDataSource(args.initiator, baudrate=args.baudrate, rtscts=args.rtscts)
        reflector_source = UartDataSource(args.reflector, baudrate=args.baudrate, rtscts=args.rtscts)

        initiator_source.set_stop_event(stop_event)
        reflector_source.set_stop_event(stop_event)

        initiator_source.open()
        reflector_source.open()

        print("Sending reboot command to initiator and reflector...")
        initiator_source.send(b'r')
        reflector_source.send(b'r')
        time.sleep(1)

        initiator_source.flush_input()
        reflector_source.flush_input()
        print("Buffers flushed.")

        if args.log_uart and session_logger:
            initiator_source.enable_logging(session_logger.initiator_log_path)
            reflector_source.enable_logging(session_logger.reflector_log_path)
            print(f"Raw logging enabled:")
            print(f"  Initiator: {session_logger.initiator_log_path}")
            print(f"  Reflector: {session_logger.reflector_log_path}")

        print("Sending start command to initiator...")
        initiator_source.send(b's')

    else:
        print("Mode: Reading from log files")
        initiator_source = FileDataSource(args.initiator)
        reflector_source = FileDataSource(args.reflector)

    # Initialize background DSP worker
    _wls = WeightedLSEstimator()
    dsp_worker = DSPWorker(
        estimators=[
            PhaseSlopeEstimator(),
            IFFTEstimator(),
            MUSICEstimator(),
            _wls,
            EMAEstimator(_wls, alpha=0.2),   # smoothed WLS — tune alpha as needed
        ]
    )

    def shutdown():
        """Signal all threads to stop and close open data sources."""
        stop_event.set()
        initiator_source.close()
        reflector_source.close()
        dsp_worker.stop()
        if session_logger:
            session_logger.close()

    viewer = launch_viewer(dark_mode=args.dark_mode, ml=args.ml, ml_handler=args.ml_handler, on_close=shutdown)
    viewer.set_dsp_queues(dsp_worker.input_queue, dsp_worker.output_queue)

    # Wire session logger into the DSP drain so every result bundle is saved.
    if session_logger:
        viewer.set_session_logger(session_logger)

    def _sigint_handler(sig, frame):
        shutdown()
        viewer.root.quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    initiator_producer = Thread(
        target=producer_worker,
        args=(initiator_source, initiator_queue, stop_event),
        kwargs={
            'status_callback': viewer.update_connection_status,
            'capabilities_callback': viewer.update_capabilities_text,
            'procedure_params_callback': viewer.update_procedure_params,
        },
        name="InitiatorProducer",
        daemon=True,
    )

    reflector_producer = Thread(
        target=producer_worker,
        args=(reflector_source, reflector_queue, stop_event),
        name="ReflectorProducer",
        daemon=True,
    )

    consumer = Thread(
        target=dual_stream_consumer,
        args=(initiator_queue, reflector_queue, viewer.update_live_data),
        name="Consumer",
        daemon=True,
    )

    print("Starting data processing pipeline...")
    dsp_worker.start()
    initiator_producer.start()
    reflector_producer.start()
    consumer.start()

    viewer.run()

    shutdown()
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
