/*
 * KNN search over the Matching Database, in C++ (plan section 13.3 step 4).
 *
 * Runs every 10 frames, not every frame, so its cost is a spike rather than a
 * constant, and timing must report both.
 *
 * To implement:
 *   - Brute force with Holden's bounding box acceleration, or the nanoflann
 *     library. Brute force first, since it is the honest reference.
 *   - Load the Matching Database, its normalization and its feature weights
 *     from the same manifest the Python side writes.
 *   - Return the top K rows and distances, with the clip boundaries respected
 *     so a retrieved row can be followed forward a frame.
 */
