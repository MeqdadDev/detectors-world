# Landmark Maps

Every landmark number used by Detectors World, in one place. Use these numbers as indexes into results, such as `hands[0][8]`.

## Hand

21 landmarks for each hand, numbered the same for left and right.

![Map of the 21 hand landmarks](../assets/maps/hand.svg){ .dw-map .dw-map--hand }

--8<-- "docs/.snippets/hand-landmarks.md"

## Pose

33 landmarks for each body. "Left" and "right" are the person's own sides, so odd numbers are on their left and even numbers on their right.

![Map of the 33 pose landmarks](../assets/maps/pose.svg){ .dw-map .dw-map--pose }

--8<-- "docs/.snippets/pose-landmarks.md"

## Face

6 key points for each face. "Left" and "right" are the person's own sides.

![Map of the 6 face key points](../assets/maps/face.svg){ .dw-map .dw-map--face }

--8<-- "docs/.snippets/face-keypoints.md"

## Face mesh

478 landmarks for each face. The map labels the most useful ones and colors each region. "Left" and "right" are the person's own sides.

![Map of the face mesh with key landmarks labeled](../assets/maps/face-mesh.svg){ .dw-map .dw-map--mesh }

### Landmarks by region

--8<-- "docs/.snippets/face-mesh-regions.md"

To see every number on your own face, use [`show_ids=True`](../detectors/face-mesh.md#find-any-landmark).
