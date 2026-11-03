"""
BVH parsing and writing round trip.

To implement
------------
- Parse a BVH file, write it back out, parse again, and require the skeleton,
  frame time and channel values to match.
- Require FK joint positions to agree through the round trip, which catches
  channel order bugs that a value by value comparison can miss.
- Cover the file variants actually used: LaFAN1 resolved, 100STYLE, and a BVH
  written by the AMASS pipeline.
"""
