#!/usr/bin/env python3
"""
3D Extractor Server Runner

Usage:
    python run.py              # Development mode with auto-reload
    python run.py --prod       # Production mode
    python run.py --port 8080  # Custom port
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="3D Extractor Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--prod", action="store_true", help="Production mode (no auto-reload)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (prod only)")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     3D Extractor Server                      ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                              ║
║  Mode: {'Production' if args.prod else 'Development (auto-reload)'}                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.prod:
        # Production mode
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info"
        )
    else:
        # Development mode with auto-reload
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="debug"
        )


if __name__ == "__main__":
    main()

