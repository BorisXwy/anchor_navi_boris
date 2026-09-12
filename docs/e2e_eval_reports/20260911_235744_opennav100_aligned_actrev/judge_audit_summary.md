# Judge audit summary: `outputs/e2e_eval/20260911_235744_opennav100_aligned_actrev`

- raw judge calls: 467 over 96 episode dirs (0 crashed, no trajectory.json); status {'completed': 167, 'unknown': 300}
- full-fidelity judged edges: 467 over 96 episodes; status {'completed': 167, 'unknown': 300}; independent verdict available for 467

## Raw calls by form (all episodes)
| form | n | completed | unknown | unknown % | conf completed | conf unknown | turn form w/o turn cmds |
|---|---|---|---|---|---|---|---|
| STOP_WAIT | 98 | 12 | 86 | 88% | 0.713 | 0.617 | 0 |
| EXIT_REGION | 65 | 29 | 36 | 55% | 0.729 | 0.597 | 0 |
| TRAVERSE_PORTAL_REGION | 44 | 17 | 27 | 61% | 0.718 | 0.607 | 0 |
| ENTER_REGION | 40 | 16 | 24 | 60% | 0.775 | 0.6 | 0 |
| ADVANCE_STRAIGHT | 33 | 9 | 24 | 73% | 0.689 | 0.6 | 0 |
| TURN_LEFT | 33 | 17 | 16 | 48% | 0.715 | 0.6 | 0 |
| PASS_LANDMARK | 31 | 13 | 18 | 58% | 0.7 | 0.597 | 0 |
| TURN_RIGHT | 28 | 18 | 10 | 36% | 0.728 | 0.6 | 0 |
| OTHER | 21 | 4 | 17 | 81% | 0.7 | 0.606 | 0 |
| FOLLOW_PATH_BOUNDARY | 15 | 3 | 12 | 80% | 0.7 | 0.6 | 0 |
| CROSS_SPACE | 15 | 8 | 7 | 47% | 0.706 | 0.6 | 0 |
| APPROACH_LANDMARK | 13 | 10 | 3 | 23% | 0.785 | 0.6 | 0 |
| VERTICAL_UP | 8 | 3 | 5 | 62% | 0.7 | 0.6 | 0 |
| BETWEEN_OBJECTS | 6 | 2 | 4 | 67% | 0.7 | 0.6 | 0 |
| TURN_AROUND | 4 | 4 | 0 | 0% | 0.812 |  | 0 |
| SELECT_PORTAL | 4 | 2 | 2 | 50% | 0.725 | 0.6 | 0 |
| CIRCUMNAVIGATE | 4 | 0 | 4 | 100% |  | 0.6 | 0 |
| TURN_TO_LANDMARK | 3 | 0 | 3 | 100% |  | 0.6 | 0 |
| VERTICAL_DOWN | 2 | 0 | 2 | 100% |  | 0.6 | 0 |

## Raw unknown reasons by category
| category | turn forms | other forms | total |
|---|---|---|---|
| crossed_into_next_stage | 0 | 1 | 1 |
| insufficient_evidence | 3 | 27 | 30 |
| landmark_absent_or_wrong_place | 8 | 59 | 67 |
| missing_turn_evidence | 0 | 5 | 5 |
| not_there_yet | 15 | 161 | 176 |
| reversed_or_stationary | 0 | 11 | 11 |
| uncategorized | 0 | 10 | 10 |

## Turn forms: structural turn-command check (raw)
| subset | n | completed | unknown |
|---|---|---|---|
| no turn command in edge | 0 | 0 | 0 |
| at least one turn command | 65 | 39 | 26 |

## Full fidelity: online status x independent verdict
| online status | independent says completed | geometry gate | n |
|---|---|---|---|
| completed | False | False | 11 |
| completed | False | True | 3 |
| completed | True | False | 71 |
| completed | True | True | 82 |
| unknown | False | False | 188 |
| unknown | False | True | 47 |
| unknown | True | False | 40 |
| unknown | True | True | 25 |

verdict classes: {"completed_confirmed": 153, "unknown_confirmed": 235, "unknown_but_verified": 65, "completed_refuted": 14}

## Full fidelity by form
| form | n | online completed | online unknown | completed confirmed | completed refuted | unknown but verified | unknown confirmed | geometry gate passed | median hop dist to point (m) |
|---|---|---|---|---|---|---|---|---|---|
| STOP_WAIT | 98 | 12 | 86 | 11 | 1 | 8 | 78 | 13 | 0.956 |
| EXIT_REGION | 65 | 29 | 36 | 26 | 3 | 3 | 33 | 25 | 1.05 |
| TRAVERSE_PORTAL_REGION | 44 | 17 | 27 | 16 | 1 | 4 | 23 | 19 | 1.036 |
| ENTER_REGION | 40 | 16 | 24 | 15 | 1 | 3 | 21 | 9 | 0.967 |
| ADVANCE_STRAIGHT | 33 | 9 | 24 | 9 | 0 | 8 | 16 | 16 | 0.772 |
| TURN_LEFT | 33 | 17 | 16 | 17 | 0 | 8 | 8 | 9 | 0.993 |
| PASS_LANDMARK | 31 | 13 | 18 | 12 | 1 | 8 | 10 | 17 | 0.961 |
| TURN_RIGHT | 28 | 18 | 10 | 18 | 0 | 9 | 1 | 11 | 0.793 |
| OTHER | 21 | 4 | 17 | 4 | 0 | 1 | 16 | 7 | 0.841 |
| FOLLOW_PATH_BOUNDARY | 15 | 3 | 12 | 3 | 0 | 2 | 10 | 7 | 0.665 |
| CROSS_SPACE | 15 | 8 | 7 | 5 | 3 | 2 | 5 | 6 | 0.738 |
| APPROACH_LANDMARK | 13 | 10 | 3 | 9 | 1 | 2 | 1 | 8 | 0.733 |
| VERTICAL_UP | 8 | 3 | 5 | 0 | 3 | 0 | 5 | 2 | 1.469 |
| BETWEEN_OBJECTS | 6 | 2 | 4 | 2 | 0 | 2 | 2 | 4 | 0.781 |
| TURN_AROUND | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 3 | 0.736 |
| SELECT_PORTAL | 4 | 2 | 2 | 2 | 0 | 0 | 2 | 0 | 0.866 |
| CIRCUMNAVIGATE | 4 | 0 | 4 | 0 | 0 | 4 | 0 | 1 | 0.838 |
| TURN_TO_LANDMARK | 3 | 0 | 3 | 0 | 0 | 0 | 3 | 0 | 1.353 |
| VERTICAL_DOWN | 2 | 0 | 2 | 0 | 0 | 1 | 1 | 0 | 0.52 |

## Last-stage judgments and STOP-blocking unknowns
- last-stage judgments: 96; unknown: 82; unknown within 3.0 m of goal (xz): 28
| episode | target | form | online | dist to goal xz (m) | independent completed | geometry gate | online reason |
|---|---|---|---|---|---|---|---|
| 166 | 1 | STOP_WAIT | completed | 0.85 | True | True | The robot approached the lamp and stopped with the lamp visible close and to the side, satisfying the 'Stop by the lamp' relation. |
| 187 | 2 | STOP_WAIT | completed | 0.87 | True | True | The robot moved forward toward the large tub, and the current panorama shows the large tub centered ahead and close with clear floor between the camera and the  |
| 11 | 1 | STOP_WAIT | completed | 1.11 | True | True | The agent moved forward through the archway and the current view shows it has passed through and is now in the adjacent room, satisfying the 'wait the archway'  |
| 1117 | 1 | APPROACH_LANDMARK | completed | 2.43 | True | True | The robot moved forward along the pool and the bar counter is now directly in front of it, matching the completion cue. |
| 721 | 6 | TURN_RIGHT | completed | 2.48 | True | False | The robot executed a right turn and then moved forward into a new area. The current panorama shows a kitchen island/counter prominently, and the view is oriente |
| 586 | 3 | STOP_WAIT | completed | 3.32 | True | False | The agent moved forward through the doorway into the room with the tile floor and stopped, satisfying the 'wait' sub-instruction to terminate at the stated spat |
| 755 | 9 | STOP_WAIT | completed | 5.87 | True | False | The robot turned right to face the staircase, then moved forward through the chronological keyframes. The current panorama shows the staircase clearly visible a |
| 1139 | 6 | STOP_WAIT | completed | 6.91 | True | False | The camera moved forward and turned right, and the current panorama shows the sink clearly visible in the foreground with the camera stationary, satisfying the  |
| 403 | 4 | STOP_WAIT | completed | 7.43 | True | False | The robot moved forward through a hallway and the current panorama shows the dining room entrance with the dining room visible ahead, matching the completion cu |
| 454 | 7 | STOP_WAIT | completed | 8.26 | True | False | The agent turned left to face the sink and then moved forward, ending with the sink centered in the view at close range, satisfying the 'wait in front of the si |
| 469 | 3 | STOP_WAIT | completed | 8.43 | True | False | The robot navigated toward the glass info panes and the current panorama shows the panes clearly visible and within a few steps, satisfying the 'wait by' relati |
| 1056 | 2 | STOP_WAIT | completed | 9.84 | False | False | The robot moved forward through the doorway, and the current view shows the doorway frame directly ahead with the floor before it unobstructed, matching the sto |
| 546 | 5 | STOP_WAIT | completed | 13.38 | True | False | The robot turned left toward the closet area and then moved forward, ending with the mirror sliding closet doors clearly visible and close in the current panora |
| 140 | 5 | STOP_WAIT | completed | 15.98 | True | False | The robot turned right toward the doorway left of the staircase, moved forward through the door frame, and the current view shows the camera has crossed into th |
| 810 | 8 | STOP_WAIT | unknown | 0.09 | False | False | The camera moved forward through the doorway but the current panorama still shows the bedroom interior (bed, carpet) prominently in the forward view, indicating |
| 7 | 3 | STOP_WAIT | unknown | 0.6 | False | True | The robot moved forward through a doorway/hallway, but the current view does not clearly show the robot standing immediately adjacent to the corner of the bar.  |
| 362 | 4 | STOP_WAIT | unknown | 0.72 | False | True | The active sub-instruction requires waiting at an open door on the left. The previous panorama showed an open door on the left, but the current panorama shows t |
| 362 | 3 | STOP_WAIT | unknown | 0.81 | False | True | The active sub-instruction requires waiting at an open door on the left. The robot turned right and moved forward, and the current panorama shows open doors on  |
| 810 | 4 | STOP_WAIT | unknown | 0.83 | True | False | The camera has moved forward through the hallway, but the RGB evidence does not clearly show that the camera has exited the bedroom door and stopped just outsid |
| 526 | 6 | STOP_WAIT | unknown | 0.9 | False | False | The robot turned right toward the door and then moved forward, but the current view shows the open white door still at a distance and off to the side rather tha |
| 259 | 4 | STOP_WAIT | unknown | 1.0 | False | False | The active sub-instruction requires waiting at the white rug in the hallway, with the white rug clearly visible in the lower foreground or immediately ahead. Th |
| 1106 | 2 | STOP_WAIT | unknown | 1.12 | False | True | The robot has moved forward through the kitchen area, but the active sub-instruction requires stopping at the refrigerator. The current RGB view shows kitchen c |
| 810 | 7 | STOP_WAIT | unknown | 1.16 | True | False | The robot moved forward through a hallway, but the RGB evidence does not clearly show the camera exiting the bedroom door and stopping just outside it. The curr |
| 810 | 6 | STOP_WAIT | unknown | 1.2 | False | False | The active sub-instruction requires stopping just outside the bedroom door after exiting the bedroom. The RGB evidence shows the camera moving through a hallway |
| 259 | 2 | STOP_WAIT | unknown | 1.27 | False | False | The robot moved forward through the hallway, but the current view shows a closet/storage area with shelves and a window, not the white rug in the hallway. The w |
| 1106 | 5 | STOP_WAIT | unknown | 1.31 | False | False | The active sub-instruction requires stopping at a refrigerator, but no refrigerator is detected in the current RGB evidence. The camera moved forward toward a c |
| 586 | 1 | STOP_WAIT | unknown | 1.33 | True | True | The active sub-instruction is a STOP_WAIT form requiring the agent to terminate at a stated spatial relation to a landmark/region. The RGB evidence shows the ag |
| 259 | 1 | STOP_WAIT | unknown | 1.41 | False | True | The active sub-instruction requires waiting at the white rug in the hallway, with the white rug clearly visible in the lower foreground or immediately ahead. Th |
| 7 | 4 | STOP_WAIT | unknown | 1.61 | False | False | The robot has approached and is now adjacent to the bar counter, but the active sub-instruction requires stopping at the corner of the bar. The current view sho |
| 7 | 2 | STOP_WAIT | unknown | 1.75 | False | True | The robot moved forward through the environment but the RGB evidence does not clearly show the robot standing immediately adjacent to the corner of the bar. The |
| 259 | 3 | STOP_WAIT | unknown | 1.83 | False | False | The white rug is detected in the current panorama, but the commanded motion was a series of left turns followed by three forward steps, and the chronological ke |
| 526 | 5 | STOP_WAIT | unknown | 2.1 | False | False | The active sub-instruction requires stopping at a safe distance in front of an open white door, with the door filling a significant portion of the forward view. |
| 1106 | 4 | STOP_WAIT | unknown | 2.17 | False | True | The robot has moved through the kitchen and is now near a refrigerator, but the final stop at a safe offset in front of the refrigerator is not clearly confirme |
| 526 | 2 | STOP_WAIT | unknown | 2.19 | False | False | The robot moved forward for 9 steps, but the RGB evidence does not show the open white door filling a significant portion of the forward view or being close eno |
| 1301 | 6 | STOP_WAIT | unknown | 2.2 | False | False | The robot has turned toward and moved forward toward a doorway, but the RGB evidence does not confirm arrival at the 'doors on the far side' at a safe stopping  |
| 321 | 5 | STOP_WAIT | unknown | 2.28 | False | False | The robot moved forward through an attic/bedroom area, but the RGB evidence does not clearly show the camera standing inside a bathroom. The 'bathroom' detector |
| 259 | 5 | STOP_WAIT | unknown | 2.29 | False | False | The robot turned left extensively and then moved forward, but the current panorama does not clearly show the white rug in the lower foreground or immediately ah |
| 526 | 3 | STOP_WAIT | unknown | 2.42 | False | False | The robot has not reached the open white door; the door remains a small, distant detection in the forward view and the commanded motion (turns and short forward |
| 821 | 9 | STOP_WAIT | unknown | 2.47 | False | False | The active sub-instruction requires stopping by the sink at a safe distance. The current RGB panorama contains a weak 'sink' detection (score 0.38) only in the  |
| 1106 | 3 | STOP_WAIT | unknown | 2.64 | False | False | The active sub-instruction requires stopping at a refrigerator, but the current RGB panorama and detector evidence show no refrigerator detection; the scene is  |
| 362 | 6 | STOP_WAIT | unknown | 2.77 | False | False | The robot has moved forward down the corridor, but the active sub-instruction requires waiting at an open door on the left. The current panorama shows an open d |
| 321 | 4 | STOP_WAIT | unknown | 2.82 | False | False | The active sub-instruction is to stop in the bathroom. The previous and current RGB panoramas show a bedroom/attic space with a bed, dresser, and wooden beams;  |
| 821 | 7 | STOP_WAIT | unknown | 3.2 | False | False | The active sub-instruction requires stopping by the sink at a safe distance. The current RGB evidence shows a weak 'counter sink' detection in the left/front-le |
| 321 | 3 | STOP_WAIT | unknown | 3.26 | False | False | The current panorama still shows a bedroom/loft interior with beds, cabinets, and wooden beams; no bathroom fixtures (toilet, sink, shower) or tiled bathroom fl |
| 1301 | 5 | STOP_WAIT | unknown | 3.42 | False | False | The active sub-instruction requires stopping at the doors on the far side of the room, at a safe distance in front of them. The current RGB evidence shows door- |
| 748 | 5 | STOP_WAIT | unknown | 3.6 | False | False | The active sub-instruction requires stopping by the dining room table at a safe offset on walkable floor. The current RGB evidence shows a dining room table det |
| 748 | 7 | STOP_WAIT | unknown | 3.69 | False | True | The active sub-instruction requires stopping by the dining room table at a safe offset on walkable floor. The current RGB evidence shows a dining room table det |
| 1301 | 2 | STOP_WAIT | unknown | 3.73 | False | False | The robot moved forward and turned slightly, but the RGB evidence does not show the camera reaching a safe stopping position directly in front of the doors on t |
| 821 | 5 | STOP_WAIT | unknown | 3.83 | False | False | The active sub-instruction is to stop by the sink, but no sink is detected in the current RGB panorama. The robot moved forward through a hallway and the curren |
| 810 | 5 | STOP_WAIT | unknown | 3.89 | True | False | The robot moved forward through a hallway and turned left, but the RGB evidence does not clearly show the camera exiting the bedroom door and stopping just outs |
| 321 | 7 | STOP_WAIT | unknown | 4.13 | False | False | The current panorama shows a strong 'bathroom' detector label covering most views, but the visible scene is still the same rustic attic/bedroom with beds, chair |
| 7 | 6 | STOP_WAIT | unknown | 4.14 | False | False | The robot has moved forward along the bar counter, but the RGB evidence does not clearly show the robot standing at the corner of the bar with the corner immedi |
| 1301 | 3 | STOP_WAIT | unknown | 4.33 | False | False | The active sub-instruction requires stopping at the doors on the far side of the room, with the doors directly ahead and close, and floor visible between the ca |
| 748 | 8 | STOP_WAIT | unknown | 4.43 | False | False | The active sub-instruction requires stopping by the dining room table at a safe offset on walkable floor. The current RGB evidence shows a 'dining room table' d |
| 748 | 4 | STOP_WAIT | unknown | 4.44 | False | False | The robot has moved toward the dining area and the table is visible, but the final spatial relation of stopping adjacent to the dining room table at a safe offs |
| 1301 | 4 | STOP_WAIT | unknown | 4.53 | False | False | The active sub-instruction requires stopping at the doors on the far side of the room, with the doors directly ahead and close, and floor visible between the ca |
| 821 | 8 | STOP_WAIT | unknown | 4.69 | False | False | The active sub-instruction requires stopping by the sink, but no sink is detected in the current RGB panorama and the camera is still moving forward. The observ |
| 1142 | 8 | STOP_WAIT | unknown | 4.71 | False | True | The robot turned right toward the doorway and then moved forward, but the current panorama still shows the doorway/door frame off to the side rather than surrou |
| 748 | 6 | STOP_WAIT | unknown | 4.89 | False | False | The active sub-instruction requires stopping by the dining room table at a safe offset on walkable floor. The current RGB evidence shows a 'dining room table' d |
| 1117 | 0 | APPROACH_LANDMARK | unknown | 5.06 | True | True | The robot moved forward alongside the pool, but the bar counter is not clearly visible directly ahead in the current view; the pool remains alongside, so the fu |
| 1142 | 4 | STOP_WAIT | unknown | 5.53 | False | False | The robot moved forward through the bedroom but the current panorama shows it is inside the bedroom with the doorway behind/around it, not clearly stopped withi |
| 1142 | 7 | STOP_WAIT | unknown | 5.83 | False | False | The robot moved forward through the bedroom but the current view shows the camera inside the bedroom facing the bed and windows, not positioned within a doorway |
| 1142 | 6 | STOP_WAIT | unknown | 5.88 | False | False | The robot turned left toward the doorway and then moved forward, but the current panorama still shows the doorway and door frame off to the side rather than sur |
| 1142 | 5 | STOP_WAIT | unknown | 6.08 | False | False | The robot moved forward toward the doorway, but the current view still shows the doorway ahead rather than the camera being positioned within the door frame wit |
| 821 | 6 | STOP_WAIT | unknown | 6.69 | False | False | The active sub-instruction requires stopping by a sink, but no sink is detected in the current RGB panorama. The robot moved forward through a hallway and into  |
| 755 | 7 | STOP_WAIT | unknown | 7.58 | False | False | The robot has moved forward through the kitchen area, but the RGB evidence does not clearly show the camera positioned on free walkable floor adjacent to the st |
| 824 | 14 | STOP_WAIT | unknown | 7.6 | True | False | The active sub-instruction is a STOP_WAIT form requiring the agent to terminate at a stated spatial relation to a landmark/region. The RGB evidence shows the ag |
| 513 | 6 | STOP_WAIT | unknown | 7.77 | False | False | The robot has moved forward through a doorway into a hallway, but the active sub-instruction requires stopping in the area between two white sofas next to the d |
| 13 | 8 | STOP_WAIT | unknown | 7.9 | False | False | The active sub-instruction requires stopping near the sink at a safe distance with the sink clearly visible. The current RGB detector evidence includes a low-co |
| 13 | 6 | STOP_WAIT | unknown | 8.07 | False | False | The robot moved forward through a living area, but the sink remains only a small, low-confidence detection and the camera has not clearly stopped at a safe offs |
| 755 | 5 | STOP_WAIT | unknown | 8.11 | False | False | The active sub-instruction requires stopping by the staircase at a safe distance on free floor. The current RGB evidence shows staircase detections in several v |
| 755 | 8 | STOP_WAIT | unknown | 8.48 | False | False | The camera has moved forward toward the staircase area, but the final stop condition is not clearly satisfied. The staircase is visible in the current panorama, |
| 13 | 5 | STOP_WAIT | unknown | 8.56 | False | False | The active sub-instruction requires stopping near the sink at a safe distance with the sink clearly visible. The current RGB panorama shows a sink detection onl |
| 755 | 6 | STOP_WAIT | unknown | 8.69 | False | False | The staircase is visible in the scene, but the camera has not clearly stopped at a safe offset on free floor beside it; the final forward motion and lack of a c |
| 513 | 2 | STOP_WAIT | unknown | 8.71 | False | False | The robot has moved forward into a dining room area, but the active sub-instruction requires stopping in the open floor space between two white sofas, adjacent  |
| 411 | 1 | STOP_WAIT | unknown | 8.8 | False | False | The robot has moved forward through the environment, and the current view shows a living room with a sofa and end table, but the required spatial relation of st |
| 513 | 5 | STOP_WAIT | unknown | 9.1 | False | False | The current panorama shows a dining area with a table and chairs, and a sofa is detected in one view, but the required spatial relation of being between two whi |
| 824 | 10 | STOP_WAIT | unknown | 9.19 | True | False | The active sub-instruction is a STOP_WAIT form requiring the agent to terminate at a stated spatial relation to a landmark/region. The RGB evidence shows the ag |
| 411 | 2 | STOP_WAIT | unknown | 9.22 | False | False | The robot has moved forward and turned, and the end table is visible in the current view, but the required spatial relation is not fully verified. The bookshelf |
| 411 | 3 | STOP_WAIT | unknown | 9.95 | False | False | The robot has moved forward and turned left, and the current view shows an end table and couch, but the required spatial relation (standing next to the end tabl |
| 804 | 12 | STOP_WAIT | unknown | 10.37 | False | False | The robot moved forward through a kitchen area, but the current view shows a kitchen island/counter rather than a dining room table. The active sub-instruction  |
| 13 | 7 | STOP_WAIT | unknown | 10.52 | False | False | The active sub-instruction requires stopping near a sink, but no sink is detected in the current RGB panorama or the chronological keyframes. The robot moved fo |
| 804 | 8 | STOP_WAIT | unknown | 12.07 | False | False | The robot moved forward through a hallway/living area, but the current view shows a fireplace and hallway rather than a dining room table directly ahead at a sa |
| 546 | 4 | STOP_WAIT | unknown | 12.25 | False | False | The robot turned right and moved forward, but the current view shows a hallway with white doors and a shelving unit, not clearly the mirror sliding closet doors |
| 513 | 3 | STOP_WAIT | unknown | 12.42 | False | False | The robot moved forward through a large room but the current view does not clearly show the robot positioned between two white sofas adjacent to the dining room |
| 804 | 10 | STOP_WAIT | unknown | 12.51 | False | False | The current panorama shows a dining room table detected in several views, but the camera does not appear to be stopped in front of it at a safe distance. The ta |
| 804 | 11 | STOP_WAIT | unknown | 12.6 | False | False | The robot moved forward through a hallway toward a wooden door, but the current view shows a closed door and kitchen cabinets rather than a dining room table at |
| 265 | 7 | STOP_WAIT | unknown | 12.63 | False | False | The robot has moved forward and turned, and the current panorama shows a hallway with a wooden door and a staircase, but the bathroom landmark is not clearly id |
| 824 | 11 | STOP_WAIT | unknown | 12.69 | True | False | The active sub-instruction is a STOP_WAIT command requiring the agent to terminate at a specific spatial relation to a landmark. The RGB evidence shows continuo |
| 546 | 3 | STOP_WAIT | unknown | 12.9 | False | False | The robot moved forward three steps, but the RGB evidence does not show the camera arriving at a safe offset next to the mirror sliding closet doors. The curren |
| 804 | 9 | STOP_WAIT | unknown | 15.03 | False | False | The robot has moved forward through the environment, but the RGB evidence does not clearly show the camera positioned in front of the dining room table at a saf |
| 265 | 6 | STOP_WAIT | unknown | 15.54 | False | False | The robot moved forward through a doorway into a room with wooden shelves and cabinets. While the current detector labels the entire view as 'bathroom' with hig |
| 265 | 8 | STOP_WAIT | unknown | 15.66 | False | False | The active sub-instruction requires waiting at the bathroom, but the current RGB evidence shows the robot still surrounded by wooden cabinetry and shelving, wit |
| 140 | 2 | STOP_WAIT | unknown | 18.67 | True | False | The robot moved forward through a doorway into an interior room, but the active sub-instruction requires taking the doorway directly left of the staircase and w |
| 140 | 4 | STOP_WAIT | unknown | 19.1 | False | False | The robot turned left and moved forward, but the RGB evidence does not clearly show it has passed through the doorway directly left of the staircase. The curren |
| 140 | 3 | STOP_WAIT | unknown | 19.71 | False | False | The robot moved forward through the environment, but the RGB evidence does not clearly show that it has passed through the doorway directly left of the staircas |

## Episodes with trajectory.json
| episode | stages | targets | completed online | end reason | STOP | success | start->final geodesic (m) | hops |
|---|---|---|---|---|---|---|---|---|
| 7 | 3 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 10.45 -> 1.75 | 0:PASS_LANDMARK:completed | 1:BETWEEN_OBJECTS:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:rgb_forward_stall | 2:STOP_WAIT:unknown |
| 13 | 4 | 9 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 12.04 -> 8.56 | 0:PASS_LANDMARK:completed | 1:APPROACH_LANDMARK:rgb_forward_stall | 1:APPROACH_LANDMARK:completed | 2:PASS_LANDMARK:unknown | 2:PASS_LANDMARK:completed | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown |
| 42 | 6 | 4 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 14.96 -> 15.53 | 0:APPROACH_LANDMARK:completed | 1:FOLLOW_PATH_BOUNDARY:unknown | 1:FOLLOW_PATH_BOUNDARY:completed | 2:TURN_RIGHT:unknown |
| 70 | 6 | 10 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 15.64 -> 16.00 | 0:TRAVERSE_PORTAL_REGION:completed | 1:TRAVERSE_PORTAL_REGION:completed | 2:TRAVERSE_PORTAL_REGION:max_steps | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:completed | 3:TRAVERSE_PORTAL_REGION:unknown | 3:TRAVERSE_PORTAL_REGION:unknown | 3:TRAVERSE_PORTAL_REGION:unknown | 3:TRAVERSE_PORTAL_REGION:unknown |
| 116 | 5 | 2 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 5.63 -> 5.01 | 0:ADVANCE_STRAIGHT:completed | 1:EXIT_REGION:rgb_navigation_cluster_lost |
| 150 | 5 | 1 | 0 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit rear direction gate | False | False | 8.75 -> 9.39 | 0:TURN_AROUND:rgb_forward_stall |
| 166 | 2 | 2 | 2 | instruction_sequence_complete | True | True | 6.01 -> 0.85 | 0:ADVANCE_STRAIGHT:completed | 1:STOP_WAIT:completed |
| 176 | 8 | 17 | 5 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.97 -> 5.15 | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:completed | 1:TURN_RIGHT:completed | 2:APPROACH_LANDMARK:unknown | 2:APPROACH_LANDMARK:unknown | 2:APPROACH_LANDMARK:completed | 3:TRAVERSE_PORTAL_REGION:completed | 4:TRAVERSE_PORTAL_REGION:unknown | 4:TRAVERSE_PORTAL_REGION:unknown | 4:TRAVERSE_PORTAL_REGION:completed | 5:TRAVERSE_PORTAL_REGION:unknown | 5:TRAVERSE_PORTAL_REGION:unknown | 5:TRAVERSE_PORTAL_REGION:unknown | 5:TRAVERSE_PORTAL_REGION:rgb_forward_stall | 5:TRAVERSE_PORTAL_REGION:rgb_navigation_cluster_lost |
| 187 | 3 | 3 | 3 | instruction_sequence_complete | True | True | 8.80 -> 0.87 | 0:PASS_LANDMARK:completed | 1:APPROACH_LANDMARK:completed | 2:STOP_WAIT:completed |
| 191 | 6 | 1 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit rear direction gate | False | False | 10.15 -> 6.60 | 0:EXIT_REGION:completed |
| 218 | 5 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 7.22 -> 6.92 | 0:EXIT_REGION:rgb_forward_stall | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:TURN_LEFT:unknown | 1:TURN_LEFT:unknown |
| 232 | 6 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 12.08 -> 16.68 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:VERTICAL_UP:unknown | 1:VERTICAL_UP:unknown | 1:VERTICAL_UP:completed |
| 247 | 6 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.78 -> 6.61 | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:rgb_forward_stall | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown |
| 265 | 5 | 10 | 4 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.04 -> 19.74 | 0:APPROACH_LANDMARK:completed | 1:TURN_RIGHT:completed | 2:ADVANCE_STRAIGHT:unknown | 2:ADVANCE_STRAIGHT:completed | 3:TURN_LEFT:unknown | 3:TURN_LEFT:completed | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:rgb_navigation_cluster_lost |
| 308 | 5 | 12 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.07 -> 8.67 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:ENTER_REGION:unknown | 1:ENTER_REGION:completed | 2:EXIT_REGION:unknown | 2:EXIT_REGION:completed | 3:ENTER_REGION:unknown | 3:ENTER_REGION:unknown | 3:ENTER_REGION:rgb_navigation_cluster_lost | 3:ENTER_REGION:unknown |
| 321 | 3 | 8 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 4.76 -> 2.87 | 0:ADVANCE_STRAIGHT:completed | 1:ENTER_REGION:unknown | 1:ENTER_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:rgb_forward_stall | 2:STOP_WAIT:unknown |
| 338 | 4 | 2 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 6.36 -> 6.33 | 0:ENTER_REGION:completed | 1:TURN_LEFT:rgb_forward_stall |
| 362 | 3 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.07 -> 0.89 | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:completed | 1:TURN_LEFT:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:rgb_forward_stall | 2:STOP_WAIT:unknown |
| 377 | 5 | 8 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.66 -> 6.86 | 0:EXIT_REGION:completed | 1:TURN_RIGHT:completed | 2:ADVANCE_STRAIGHT:completed | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown |
| 403 | 3 | 5 | 3 | instruction_sequence_complete | True | False | 8.81 -> 8.13 | 0:TURN_RIGHT:unknown | 0:TURN_RIGHT:completed | 1:TRAVERSE_PORTAL_REGION:unknown | 1:TRAVERSE_PORTAL_REGION:completed | 2:STOP_WAIT:completed |
| 423 | 5 | 9 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 11.14 -> 2.78 | 0:TURN_LEFT:completed | 1:CROSS_SPACE:unknown | 1:CROSS_SPACE:completed | 2:TURN_LEFT:completed | 3:CROSS_SPACE:unknown | 3:CROSS_SPACE:rgb_forward_stall | 3:CROSS_SPACE:unknown | 3:CROSS_SPACE:rgb_forward_stall | 3:CROSS_SPACE:unknown |
| 439 | 7 | 0 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.76 -> 5.76 |  |
| 454 | 6 | 8 | 6 | instruction_sequence_complete | True | False | 9.38 -> 12.70 | 0:TURN_LEFT:completed | 1:TURN_RIGHT:completed | 2:ENTER_REGION:completed | 3:TURN_LEFT:completed | 4:ENTER_REGION:unknown | 4:ENTER_REGION:unknown | 4:ENTER_REGION:completed | 5:STOP_WAIT:completed |
| 469 | 4 | 4 | 4 | instruction_sequence_complete | True | False | 16.25 -> 8.63 | 0:TURN_RIGHT:completed | 1:FOLLOW_PATH_BOUNDARY:completed | 2:PASS_LANDMARK:completed | 3:STOP_WAIT:completed |
| 513 | 2 | 7 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 4.34 -> 8.88 | 0:ENTER_REGION:unknown | 0:ENTER_REGION:completed | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:rgb_forward_stall | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown |
| 526 | 2 | 7 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.68 -> 1.75 | 0:PASS_LANDMARK:unknown | 0:PASS_LANDMARK:completed | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:rgb_forward_stall | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown |
| 546 | 3 | 6 | 3 | instruction_sequence_complete | True | False | 4.61 -> 14.75 | 0:TURN_RIGHT:completed | 1:ENTER_REGION:unknown | 1:ENTER_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:completed |
| 559 | 4 | 2 | 0 | rgb_only_physical_failure_recovery_failed | False | False | 11.79 -> 7.48 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:max_steps |
| 576 | 5 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 10.26 -> 9.92 | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:unknown |
| 602 | 5 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.77 -> 7.58 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown |
| 620 | 3 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.84 -> 4.67 | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:rgb_navigation_cluster_lost |
| 655 | 4 | 2 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 9.51 -> 6.67 | 0:EXIT_REGION:completed | 1:TURN_LEFT:unknown |
| 677 | 4 | 4 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 5.75 -> 3.14 | 0:APPROACH_LANDMARK:completed | 1:TURN_LEFT:completed | 2:TURN_LEFT:unknown | 2:TURN_LEFT:unknown |
| 705 | 2 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 10.14 -> 6.24 | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:rgb_forward_stall | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown |
| 721 | 4 | 7 | 4 | instruction_sequence_complete | True | True | 4.75 -> 2.96 | 0:TURN_AROUND:completed | 1:TURN_LEFT:completed | 2:SELECT_PORTAL:unknown | 2:SELECT_PORTAL:rgb_navigation_cluster_lost | 2:SELECT_PORTAL:unknown | 2:SELECT_PORTAL:completed | 3:TURN_RIGHT:completed |
| 743 | 6 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 10.27 -> 13.27 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:rgb_navigation_cluster_lost | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:TURN_LEFT:unknown | 1:TURN_LEFT:rgb_forward_stall |
| 755 | 4 | 10 | 4 | instruction_sequence_complete | True | False | 10.91 -> 6.18 | 0:FOLLOW_PATH_BOUNDARY:completed | 1:TURN_LEFT:rgb_forward_stall | 1:TURN_LEFT:completed | 2:TURN_RIGHT:unknown | 2:TURN_RIGHT:completed | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:completed |
| 781 | 2 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.61 -> 4.35 | 0:BETWEEN_OBJECTS:unknown | 0:BETWEEN_OBJECTS:rgb_forward_stall | 0:BETWEEN_OBJECTS:unknown | 0:BETWEEN_OBJECTS:unknown |
| 804 | 6 | 13 | 5 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 12.82 -> 12.59 | 0:EXIT_REGION:completed | 1:PASS_LANDMARK:completed | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:completed | 3:TURN_LEFT:completed | 4:ENTER_REGION:unknown | 4:ENTER_REGION:completed | 5:STOP_WAIT:unknown | 5:STOP_WAIT:unknown | 5:STOP_WAIT:unknown | 5:STOP_WAIT:unknown | 5:STOP_WAIT:unknown |
| 821 | 5 | 10 | 4 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.86 -> 4.14 | 0:EXIT_REGION:completed | 1:OTHER:completed | 2:TURN_LEFT:completed | 3:SELECT_PORTAL:rgb_navigation_cluster_lost | 3:SELECT_PORTAL:completed | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown |
| 842 | 6 | 4 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 11.80 -> 5.05 | 0:ADVANCE_STRAIGHT:completed | 1:PASS_LANDMARK:completed | 2:TURN_LEFT:unknown | 2:TURN_LEFT:unknown |
| 1056 | 3 | 3 | 3 | instruction_sequence_complete | True | False | 6.71 -> 10.91 | 0:ADVANCE_STRAIGHT:completed | 1:TURN_RIGHT:completed | 2:STOP_WAIT:completed |
| 1071 | 6 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 13.37 -> 5.04 | 0:TURN_LEFT:completed | 1:CROSS_SPACE:completed | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:unknown | 2:TRAVERSE_PORTAL_REGION:rgb_forward_stall | 2:TRAVERSE_PORTAL_REGION:rgb_forward_stall |
| 1084 | 6 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 3.85 -> 12.97 | 0:EXIT_REGION:completed | 1:ENTER_REGION:unknown | 1:ENTER_REGION:rgb_navigation_cluster_lost | 1:ENTER_REGION:unknown | 1:ENTER_REGION:unknown | 1:ENTER_REGION:unknown |
| 1087 | 5 | 1 | 0 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 5.60 -> 5.60 | 0:TURN_RIGHT:rgb_navigation_cluster_lost |
| 1106 | 3 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.15 -> 1.12 | 0:TURN_RIGHT:completed | 1:ENTER_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown |
| 1133 | 3 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 11.94 -> 8.97 | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown |
| 1142 | 4 | 9 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.57 -> 5.60 | 0:TURN_LEFT:completed | 1:PASS_LANDMARK:completed | 2:CROSS_SPACE:completed | 3:STOP_WAIT:rgb_forward_stall | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown |
| 1284 | 5 | 3 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 8.62 -> 3.82 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:TURN_LEFT:rgb_forward_stall |
| 1307 | 3 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.87 -> 4.33 | 0:TURN_RIGHT:unknown | 0:TURN_RIGHT:completed | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:rgb_forward_stall | 1:PASS_LANDMARK:rgb_forward_stall |
| 11 | 2 | 2 | 2 | instruction_sequence_complete | True | True | 7.11 -> 1.11 | 0:CROSS_SPACE:completed | 1:STOP_WAIT:completed |
| 40 | 7 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 14.96 -> 11.32 | 0:ENTER_REGION:completed | 1:OTHER:completed | 2:ENTER_REGION:unknown | 2:ENTER_REGION:unknown | 2:ENTER_REGION:rgb_forward_stall | 2:ENTER_REGION:rgb_navigation_cluster_lost | 2:ENTER_REGION:rgb_forward_stall |
| 52 | 3 | 4 | 1 | rgb_only_sequence_recovery_failed | False | False | 14.44 -> 1.32 | 0:TURN_RIGHT:completed | 1:TRAVERSE_PORTAL_REGION:unknown | 1:TRAVERSE_PORTAL_REGION:rgb_navigation_cluster_lost | 1:TRAVERSE_PORTAL_REGION:unknown |
| 94 | 4 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 13.93 -> 12.73 | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown | 0:ADVANCE_STRAIGHT:unknown |
| 140 | 3 | 6 | 3 | instruction_sequence_complete | True | False | 13.37 -> 17.25 | 0:TRAVERSE_PORTAL_REGION:completed | 1:TRAVERSE_PORTAL_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:completed |
| 156 | 7 | 5 | 3 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 11.98 -> 7.46 | 0:EXIT_REGION:completed | 1:CROSS_SPACE:completed | 2:TRAVERSE_PORTAL_REGION:rgb_forward_stall | 2:TRAVERSE_PORTAL_REGION:completed | 3:TURN_LEFT:unknown |
| 171 | 6 | 7 | 3 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 11.21 -> 17.99 | 0:CROSS_SPACE:unknown | 0:CROSS_SPACE:unknown | 0:CROSS_SPACE:completed | 1:TURN_LEFT:completed | 2:VERTICAL_UP:completed | 3:TURN_RIGHT:unknown | 3:TURN_RIGHT:unknown |
| 181 | 8 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.01 -> 8.24 | 0:VERTICAL_UP:unknown | 0:VERTICAL_UP:rgb_navigation_cluster_lost | 0:VERTICAL_UP:rgb_forward_stall | 0:VERTICAL_UP:unknown |
| 190 | 4 | 4 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 10.15 -> 2.01 | 0:PASS_LANDMARK:completed | 1:OTHER:unknown | 1:OTHER:completed | 2:TURN_RIGHT:rgb_navigation_cluster_lost |
| 207 | 3 | 8 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 11.51 -> 20.66 | 0:EXIT_REGION:max_steps | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:OTHER:unknown | 1:OTHER:rgb_forward_stall | 1:OTHER:unknown | 1:OTHER:unknown | 1:OTHER:unknown |
| 226 | 6 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.63 -> 8.32 | 0:EXIT_REGION:rgb_forward_stall | 0:EXIT_REGION:unknown | 0:EXIT_REGION:rgb_forward_stall | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown |
| 244 | 4 | 2 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 6.91 -> 4.48 | 0:TRAVERSE_PORTAL_REGION:completed | 1:TURN_LEFT:unknown |
| 259 | 2 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.77 -> 1.50 | 0:EXIT_REGION:completed | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown |
| 275 | 4 | 4 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 8.34 -> 11.87 | 0:PASS_LANDMARK:completed | 1:ENTER_REGION:completed | 2:TURN_LEFT:unknown | 2:TURN_LEFT:unknown |
| 312 | 3 | 3 | 0 | rgb_only_physical_failure_recovery_failed | False | False | 6.71 -> 2.32 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:rgb_forward_stall |
| 330 | 3 | 3 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.07 -> 9.34 | 0:VERTICAL_DOWN:unknown | 0:VERTICAL_DOWN:unknown | 0:VERTICAL_DOWN:rgb_navigation_cluster_lost |
| 348 | 5 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 11.22 -> 16.21 | 0:VERTICAL_UP:rgb_navigation_cluster_lost | 0:VERTICAL_UP:max_steps | 0:VERTICAL_UP:unknown | 0:VERTICAL_UP:rgb_forward_stall | 0:VERTICAL_UP:completed |
| 371 | 3 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.30 -> 4.46 | 0:TRAVERSE_PORTAL_REGION:completed | 1:OTHER:unknown | 1:OTHER:unknown | 1:OTHER:rgb_forward_stall | 1:OTHER:unknown |
| 387 | 8 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 10.36 -> 11.99 | 0:TURN_RIGHT:completed | 1:EXIT_REGION:unknown | 1:EXIT_REGION:max_steps | 1:EXIT_REGION:completed | 2:TURN_RIGHT:unknown | 2:TURN_RIGHT:unknown |
| 411 | 2 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.02 -> 11.41 | 0:APPROACH_LANDMARK:completed | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:unknown | 1:STOP_WAIT:rgb_forward_stall | 1:STOP_WAIT:max_steps |
| 432 | 3 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.98 -> 2.86 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:rgb_forward_stall | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown |
| 447 | 2 | 3 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 11.76 -> 6.25 | 0:BETWEEN_OBJECTS:unknown | 0:BETWEEN_OBJECTS:completed | 1:TURN_LEFT:rgb_forward_stall |
| 461 | 5 | 6 | 3 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 4.70 -> 6.54 | 0:TURN_AROUND:completed | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:completed | 2:TURN_RIGHT:completed | 3:TURN_RIGHT:unknown | 3:TURN_RIGHT:unknown |
| 479 | 4 | 4 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.33 -> 5.62 | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:unknown | 0:FOLLOW_PATH_BOUNDARY:rgb_forward_stall | 0:FOLLOW_PATH_BOUNDARY:unknown |
| 516 | 6 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 9.57 -> 7.58 | 0:PASS_LANDMARK:unknown | 0:PASS_LANDMARK:unknown | 0:PASS_LANDMARK:unknown | 0:PASS_LANDMARK:completed | 1:TRAVERSE_PORTAL_REGION:completed | 2:TURN_LEFT:rgb_forward_stall |
| 531 | 4 | 4 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 5.27 -> 7.13 | 0:EXIT_REGION:completed | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:rgb_forward_stall | 1:PASS_LANDMARK:unknown |
| 550 | 4 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 16.91 -> 17.62 | 0:OTHER:unknown | 0:OTHER:unknown | 0:OTHER:unknown | 0:OTHER:unknown | 0:OTHER:unknown |
| 568 | 3 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 10.37 -> 3.52 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown |
| 586 | 2 | 4 | 2 | instruction_sequence_complete | True | False | 5.35 -> 3.32 | 0:ENTER_REGION:completed | 1:STOP_WAIT:unknown | 1:STOP_WAIT:rgb_forward_stall | 1:STOP_WAIT:completed |
| 609 | 3 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.10 -> 6.46 | 0:EXIT_REGION:completed | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown | 1:ADVANCE_STRAIGHT:unknown |
| 643 | 3 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 10.24 -> 11.40 | 0:TURN_LEFT:completed | 1:OTHER:unknown | 1:OTHER:unknown | 1:OTHER:unknown | 1:OTHER:unknown |
| 670 | 4 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 8.82 -> 5.40 | 0:ENTER_REGION:completed | 1:TURN_LEFT:unknown | 1:TURN_LEFT:completed | 2:ENTER_REGION:unknown | 2:ENTER_REGION:unknown | 2:ENTER_REGION:unknown |
| 698 | 5 | 4 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.12 -> 7.86 | 0:EXIT_REGION:completed | 1:EXIT_REGION:unknown | 1:EXIT_REGION:rgb_forward_stall | 1:EXIT_REGION:unknown |
| 715 | 5 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 12.48 -> 10.19 | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:unknown | 0:TRAVERSE_PORTAL_REGION:completed | 1:TURN_RIGHT:rgb_forward_stall |
| 739 | 5 | 3 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 13.81 -> 11.17 | 0:EXIT_REGION:completed | 1:TRAVERSE_PORTAL_REGION:completed | 2:TURN_RIGHT:rgb_forward_stall |
| 748 | 4 | 9 | 3 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 17.01 -> 4.79 | 0:APPROACH_LANDMARK:completed | 1:TRAVERSE_PORTAL_REGION:completed | 2:TURN_RIGHT:completed | 3:STOP_WAIT:rgb_forward_stall | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown | 3:STOP_WAIT:unknown |
| 765 | 3 | 2 | 0 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit right direction gate | False | False | 6.31 -> 8.35 | 0:TURN_RIGHT:rgb_forward_stall | 0:TURN_RIGHT:rgb_forward_stall |
| 787 | 5 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 8.55 -> 3.57 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:TURN_LEFT:unknown | 1:TURN_LEFT:unknown |
| 810 | 3 | 9 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 7.01 -> 0.81 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:EXIT_REGION:unknown | 1:EXIT_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown |
| 824 | 5 | 15 | 4 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.54 -> 9.00 | 0:TURN_AROUND:completed | 1:OTHER:completed | 2:CROSS_SPACE:unknown | 2:CROSS_SPACE:rgb_forward_stall | 2:CROSS_SPACE:completed | 3:ENTER_REGION:max_steps | 3:ENTER_REGION:unknown | 3:ENTER_REGION:unknown | 3:ENTER_REGION:completed | 4:STOP_WAIT:rgb_forward_stall | 4:STOP_WAIT:unknown | 4:STOP_WAIT:unknown | 4:STOP_WAIT:rgb_forward_stall | 4:STOP_WAIT:rgb_forward_stall | 4:STOP_WAIT:unknown |
| 1051 | 5 | 2 | 2 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 10.44 -> 8.11 | 0:TURN_AROUND:completed | 1:EXIT_REGION:completed |
| 1061 | 4 | 6 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.66 -> 2.60 | 0:TURN_RIGHT:completed | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:unknown | 1:PASS_LANDMARK:rgb_forward_stall | 1:PASS_LANDMARK:unknown |
| 1077 | 5 | 6 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.79 -> 7.06 | 0:TURN_RIGHT:completed | 1:EXIT_REGION:completed | 2:ENTER_REGION:unknown | 2:ENTER_REGION:unknown | 2:ENTER_REGION:unknown | 2:ENTER_REGION:rgb_navigation_cluster_lost |
| 1085 | 6 | 2 | 1 | vlm_selection_failed: No floor-bearing candidate remains inside the explicit left direction gate | False | False | 3.85 -> 7.07 | 0:EXIT_REGION:completed | 1:TURN_LEFT:rgb_forward_stall |
| 1092 | 4 | 5 | 0 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 4.98 -> 9.97 | 0:TURN_TO_LANDMARK:unknown | 0:TURN_TO_LANDMARK:rgb_forward_stall | 0:TURN_TO_LANDMARK:unknown | 0:TURN_TO_LANDMARK:unknown | 0:TURN_TO_LANDMARK:rgb_navigation_cluster_lost |
| 1117 | 1 | 2 | 1 | instruction_sequence_complete | True | True | 8.57 -> 2.43 | 0:APPROACH_LANDMARK:unknown | 0:APPROACH_LANDMARK:completed |
| 1139 | 4 | 7 | 4 | instruction_sequence_complete | True | False | 7.59 -> 8.68 | 0:EXIT_REGION:unknown | 0:EXIT_REGION:unknown | 0:EXIT_REGION:completed | 1:CROSS_SPACE:rgb_navigation_cluster_lost | 1:CROSS_SPACE:completed | 2:ENTER_REGION:completed | 3:STOP_WAIT:completed |
| 1148 | 5 | 5 | 1 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 8.25 -> 4.74 | 0:APPROACH_LANDMARK:completed | 1:CIRCUMNAVIGATE:unknown | 1:CIRCUMNAVIGATE:unknown | 1:CIRCUMNAVIGATE:unknown | 1:CIRCUMNAVIGATE:unknown |
| 1301 | 3 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 9.89 -> 3.07 | 0:ADVANCE_STRAIGHT:completed | 1:ENTER_REGION:completed | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown | 2:STOP_WAIT:unknown |
| 1406 | 4 | 7 | 2 | vlm_selection_failed: No floor-bearing candidate can be sent to the VLM | False | False | 6.90 -> 1.74 | 0:TURN_LEFT:completed | 1:EXIT_REGION:completed | 2:PASS_LANDMARK:unknown | 2:PASS_LANDMARK:unknown | 2:PASS_LANDMARK:unknown | 2:PASS_LANDMARK:unknown | 2:PASS_LANDMARK:rgb_navigation_cluster_lost |

