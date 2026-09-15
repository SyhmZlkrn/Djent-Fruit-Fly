Drop djent tabs here for the fly to learn from:
  Guitar Pro files (.gp = GP7/8, .gp5, .gp4, .gp3)  <- best: exact rhythm, tuning, sections
  tab-only PDFs exported from Guitar Pro           <- read by the OMR; other PDF layouts may fail
Audio files are NOT used for learning (only data/rational_gaze_backing.mp3 is used, as the backing track).
Every song is transposed so its lowest open string is the fly's low F (8-string, F standard).
Then:  python -m flybrain_composer.cli corpus      (check what loads)
       python -m flybrain_composer.cli fit-multi   (one read-out for all songs)
       python -m flybrain_composer.cli generate    (riffs of its own -> output/)
       python -m flybrain_composer.cli play --generate   (live, with 👍/👎 on the stage)
