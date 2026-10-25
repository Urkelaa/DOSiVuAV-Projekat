import os

import params
import examples
import make_video


def missing_files(folder, names):
    return [name for name in names if not os.path.exists(folder + "/" + name)]


def ensure_outputs(calib):
    if len(missing_files(params.EXAMPLES_DIR, params.WRITEUP_IMAGES)) > 0:
        examples.generate_examples(calib)

    for name in missing_files(params.OUTPUT_VIDEOS_DIR, params.WRITEUP_VIDEOS):
        make_video.make_video(params.TEST_VIDEOS_DIR + "/" + name, params.OUTPUT_VIDEOS_DIR + "/" + name, calib)
