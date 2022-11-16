# Evolve - evolution simulation

A much simplified version of this rad thing I saw on YouTube [I programed some creatures. They Evolved.](https://youtu.be/N3tRFayqVtk)

Blobs are created with some combination of senses and actions. Blobs may sense and move up, down, left and right. Blobs also have a multi-layer perceptron brain to process input signals to action signals. All of these attributes are determined by an 8-bit ingeger assigned at birth.

When mating, the new Blob will inherit half of it's genes from each parent.

Blobs are periodically trained to respond more strongly to higher concentrations of Blobs nearby.

Blobs only find each other and mate. They have no other abilities or drives.

We create a square grid, Blobland, to host a population Blobs. After so many steps of moving around and mating, all Blobs not in the safe zone are killed. The remaining are dispersed throughout Blobland, and new Blobs are added to replenish the population to the initial setting.

As each epoch completes, statistics about survival and mating opportunities are printed. 

After the chosen number of epoch completes an animation of each epoch may be viewed, along with a histogram relating genetic diversity.
