#!/usr/bin/env python3
"""Export an existing sliced checkpoint as portable, checksummed 1.9 GB shards."""
import argparse
import json

from cocurve.checkpoints import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-bytes", type=int, default=1_900_000_000)
    args = parser.parse_args()
    result = export_checkpoint(args.source, args.output, max_shard_bytes=args.shard_bytes)
    print(json.dumps({"output": args.output, "tensor_bytes_preserved": result["tensor_bytes_preserved"],
                      "files": len(result["files"])}))


if __name__ == "__main__":
    main()
