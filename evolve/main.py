from settings import *
from blobland import Blobland
from datetime import datetime


def main() -> Blobland:
    """Create a Blobland of Blobs and perform a simulation."""
    try:
        blobland = Blobland()
        for epoch in range(1, WORLD_EPOCHS + 1):
            blobland.epoch_peak = 0
            blobland.manage_population()
            epoch_start_time = datetime.now()
            for step in range(1, STEPS_PER_WORLD_EPOCH + 1):
                blobs = list(blobland.blobs.values())
                for blob in blobs:
                    if step % STEPS_BETWEEN_TRAINING == 0:
                        blob.train()
                        blob.find_mate()
                    blob.update()
                    blobland.save_step_data(epoch, step, blob)
            blobland.cleanup_epoch(epoch, step, epoch_start_time)
    except KeyboardInterrupt:
        return blobland
    return blobland


if __name__ == "__main__":
    main()
