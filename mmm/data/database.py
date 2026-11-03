"""
The two databases, and the clip append that makes the KB editable.

- MatchingDatabase: one feature row per frame (n_frames, 27), with
  normalization statistics, feature weights, clip boundaries and the search
  index. Used for searching.
- AnimationDatabase: the full pose per frame (n_frames, pose_dim). Used for
  drawing the character and as the retrieved context the model reads.

To implement
------------
- Build both from a list of clips, recording for every row which clip it came
  from and its frame index inside that clip, so a retrieved row can be followed
  forward in time and so leave one clip out retrieval can exclude a source.
- append_clip(): compute the clip's features, append its rows to both
  databases, extend the index. No encoder and no retraining; this is the
  mechanism behind the main claim (plan section 4.6).
- Memory accounting, reported honestly: the full Animation Database stays
  resident, about 50 MB for locomotion, the same as classic motion matching.
  The selling point is editability, not memory.
- save / load as .npy plus a small JSON manifest (skeleton, feature layout,
  normalization, clip list, licence of each source dataset).
- next_frame_of(row): the row after a retrieved one, respecting clip
  boundaries, since the model may read candidates' next frames rather than the
  candidates themselves.

Compression, PCA or product quantization, is future work and deliberately not
implemented here.
"""
