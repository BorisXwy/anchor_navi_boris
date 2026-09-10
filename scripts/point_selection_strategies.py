#!/usr/bin/env python3
"""Executable instruction-form point-selection taxonomy."""

STRATEGIES = {
    "EXIT_REGION": {
        "perception": [
            "doorway/opening detection",
            "floor segmentation",
            "depth"
        ],
        "detect": "all portals on the boundary of the current room",
        "candidates": "floor pixels whose depth rays pass through a detected portal and land beyond its plane",
        "constraints": [
            "candidate is outside source room",
            "portal crossing is collision-free",
            "clearance from frame and door leaf"
        ],
        "rank": "instruction-matched portal, forward progress, portal-center clearance, then depth",
        "arrival": "camera center crosses the chosen portal plane",
        "fallback": "select a connected-floor opening, never ordinary floor still inside the room",
        "waypoints": "one point beyond portal; use a threshold point first only when the doorway is narrow"
    },
    "ENTER_REGION": {
        "perception": [
            "open-vocabulary room/landmark recognition",
            "doorway detection",
            "floor segmentation",
            "depth"
        ],
        "detect": "portal leading to the named destination and destination evidence visible through it",
        "candidates": "floor just inside the detected destination, beyond the portal plane",
        "constraints": [
            "destination semantics match instruction",
            "not source-side threshold floor",
            "frame clearance"
        ],
        "rank": "destination confidence, portal consistency, safe clearance, then shallow inside depth",
        "arrival": "camera is beyond portal with destination occupying the view",
        "fallback": "choose an opening only if the VLM verifies destination evidence; otherwise request a new panorama",
        "waypoints": "portal threshold plus inside-region point for occluded/narrow entrances"
    },
    "TURN_LEFT": {
        "perception": [
            "floor connectivity",
            "free-space/opening detection",
            "depth"
        ],
        "detect": "navigable corridor/doorway/open space in the left angular sector",
        "candidates": "floor in a connected opening 30-120 degrees left of incoming heading",
        "constraints": [
            "exclude wall pixels",
            "exclude left floor with no traversable continuation",
            "exclude backtracking sector"
        ],
        "rank": "angular agreement with left turn, connected free-space width, then forward depth after turn",
        "arrival": "heading aligns with and position enters selected left branch",
        "fallback": "rotate toward left opening and reacquire panorama; do not target the left wall",
        "waypoints": "one branch-entry point"
    },
    "TURN_RIGHT": {
        "perception": [
            "floor connectivity",
            "free-space/opening detection",
            "depth"
        ],
        "detect": "navigable corridor/doorway/open space in the right angular sector",
        "candidates": "floor in a connected opening 30-120 degrees right of incoming heading",
        "constraints": [
            "exclude wall pixels",
            "exclude right floor with no traversable continuation",
            "exclude backtracking sector"
        ],
        "rank": "angular agreement with right turn, connected free-space width, then forward depth after turn",
        "arrival": "heading aligns with and position enters selected right branch",
        "fallback": "rotate toward right opening and reacquire panorama; do not target the right wall",
        "waypoints": "one branch-entry point"
    },
    "TURN_AROUND": {
        "perception": [
            "incoming-heading history",
            "floor connectivity",
            "free-space detection"
        ],
        "detect": "navigable route in the reverse angular sector",
        "candidates": "connected floor 135-225 degrees from incoming heading",
        "constraints": [
            "must be reverse rather than a side branch",
            "must have traversable continuation"
        ],
        "rank": "closeness to 180-degree reversal and free-space width",
        "arrival": "agent aligns with and begins entering reverse route",
        "fallback": "perform controlled 180-degree rotation then select forward connected floor",
        "waypoints": "one reverse-route point"
    },
    "TURN_TO_LANDMARK": {
        "perception": [
            "named-landmark RGB recognition",
            "2-D landmark bearing",
            "connected floor segmentation"
        ],
        "detect": "the explicitly named landmark and the walkable floor ray aligned with its image bearing",
        "candidates": "floor anchors on the landmark bearing, including lateral or rear sectors when the landmark is there",
        "constraints": [
            "the final camera-plus-pixel ray points toward the named landmark",
            "anchor is on connected floor, never on the landmark mask or a wall",
            "do not apply an ordinary forward-route prior unless the clause also says to continue forward"
        ],
        "rank": "landmark identity and bearing, connected route continuity, safe floor clearance",
        "arrival": "heading aligns with the named landmark and the selected floor ray begins the requested turn",
        "fallback": "keep the landmark in RGB view, select the nearest same-bearing floor anchor, and request a local view when the landmark is clipped",
        "waypoints": "one bearing-aligned floor point; reacquire the landmark after the turn"
    },
    "VERTICAL_UP": {
        "perception": [
            "stair/step detection or segmentation",
            "depth",
            "estimated surface elevation"
        ],
        "detect": "ascending stair flight and its upper landing",
        "candidates": "stair tread centerline points progressing upward, preferably upper landing floor",
        "constraints": [
            "positive elevation gain",
            "avoid railing/void",
            "successive tread connectivity"
        ],
        "rank": "upper-landing confidence, elevation gain, centerline clearance",
        "arrival": "agent reaches upper landing and elevation stabilizes",
        "fallback": "select center of nearest ascending tread and replan after partial climb",
        "waypoints": "multi-point stair centerline ending at upper landing"
    },
    "VERTICAL_DOWN": {
        "perception": [
            "stair/step detection or segmentation",
            "depth",
            "estimated surface elevation"
        ],
        "detect": "descending stair flight and its lower landing",
        "candidates": "stair tread centerline points progressing downward, preferably lower landing floor",
        "constraints": [
            "negative elevation change",
            "avoid railing/void",
            "successive tread connectivity"
        ],
        "rank": "lower-landing confidence, safe tread visibility, centerline clearance",
        "arrival": "agent reaches lower landing and elevation stabilizes",
        "fallback": "select center of nearest descending tread and replan after partial descent",
        "waypoints": "multi-point stair centerline ending at lower landing"
    },
    "PASS_LANDMARK": {
        "perception": [
            "open-vocabulary object detection",
            "instance mask",
            "depth",
            "incoming route direction"
        ],
        "detect": "the noun-phrase landmark referenced by pass/past",
        "candidates": "connected floor whose route projection is beyond the detected landmark",
        "constraints": [
            "landmark projection is behind candidate along travel axis",
            "safe lateral object clearance",
            "candidate is not merely beside/before object"
        ],
        "rank": "detector-text match, positive beyond-margin, route continuity, then clearance",
        "arrival": "landmark has moved behind agent and tracked landmark bearing crosses the pass boundary",
        "fallback": "do not guess beyond without detection; approach while keeping landmark visible, then redetect",
        "waypoints": "side-clearance waypoint followed by beyond-landmark waypoint when obstacle blocks direct passage"
    },
    "CIRCUMNAVIGATE": {
        "perception": [
            "open-vocabulary obstacle detection",
            "instance mask",
            "depth",
            "side-relation parser"
        ],
        "detect": "obstacle and requested left/right side",
        "candidates": "floor in a clearance corridor on the requested side, followed by floor beyond obstacle",
        "constraints": [
            "never switch to wrong side",
            "maintain clearance from object mask",
            "second point projects beyond obstacle"
        ],
        "rank": "side correctness, collision clearance, route continuity, beyond progress",
        "arrival": "agent clears far extent of obstacle on requested side",
        "fallback": "select visible side-clearance point, preserve object track, and replan beyond point later",
        "waypoints": "normally two: lateral side point then beyond-obstacle point"
    },
    "CROSS_SPACE": {
        "perception": [
            "region/open-space segmentation",
            "far-boundary estimation",
            "depth"
        ],
        "detect": "near and far boundaries of referenced open region",
        "candidates": "free floor at or just beyond the far boundary",
        "constraints": [
            "candidate must be farther than region center",
            "connected crossing path",
            "not near-side floor"
        ],
        "rank": "far-side confidence, direct crossing alignment, free-space clearance",
        "arrival": "agent crosses the far boundary",
        "fallback": "select a deep centerline point and re-estimate far boundary after progress",
        "waypoints": "center point then far-side point when far boundary is occluded"
    },
    "BETWEEN_OBJECTS": {
        "perception": [
            "two open-vocabulary detections",
            "instance masks",
            "depth",
            "clearance estimation"
        ],
        "detect": "both referenced objects and the gap between their masks",
        "candidates": "floor centered in or just beyond the navigable gap",
        "constraints": [
            "both objects bracket candidate",
            "gap exceeds agent clearance",
            "candidate not on either instance"
        ],
        "rank": "two-object text match, midpoint relation, gap width, route progress",
        "arrival": "agent enters or clears the bracketed gap",
        "fallback": "if one object is missing, keep both-search view and defer final gap point",
        "waypoints": "gap-center point, optionally followed by point beyond gap"
    },
    "SELECT_PORTAL": {
        "perception": [
            "all-door/opening detection",
            "bearing ordering",
            "depth",
            "destination evidence"
        ],
        "detect": "all candidate portals, then resolve ordinal and left/right phrase",
        "candidates": "floor immediately through the resolved portal",
        "constraints": [
            "portal identity satisfies ordinal/side",
            "exclude adjacent portals",
            "frame clearance"
        ],
        "rank": "portal identity certainty, destination evidence, clearance",
        "arrival": "agent crosses the resolved portal",
        "fallback": "reacquire panorama until required portal set is observable; never silently choose another door",
        "waypoints": "one point through selected portal"
    },
    "TRAVERSE_PORTAL_REGION": {
        "perception": [
            "portal/region detection",
            "floor segmentation",
            "depth"
        ],
        "detect": "referenced opening or intermediate region boundaries",
        "candidates": "floor beyond the far portal/boundary",
        "constraints": [
            "must clear rather than stop in portal",
            "connected floor",
            "frame/object clearance"
        ],
        "rank": "reference match, beyond-boundary margin, free-space width",
        "arrival": "camera clears far boundary",
        "fallback": "select threshold/center point and replan beyond after crossing",
        "waypoints": "threshold plus beyond point for long/occluded traversal"
    },
    "FOLLOW_PATH_BOUNDARY": {
        "perception": [
            "hall/path/floor segmentation",
            "wall/boundary line estimation",
            "depth"
        ],
        "detect": "referenced corridor/path and requested wall/boundary relation",
        "candidates": "distant connected floor along tangent of path/boundary",
        "constraints": [
            "preserve requested side distance",
            "do not cross boundary",
            "reject side branches"
        ],
        "rank": "path tangent consistency, boundary-distance consistency, forward depth",
        "arrival": "sufficient along-path progress or next stated landmark reached",
        "fallback": "choose short tangent waypoint and repeatedly re-estimate boundary",
        "waypoints": "receding-horizon sequence along path tangent"
    },
    "ADVANCE_STRAIGHT": {
        "perception": [
            "connected floor segmentation",
            "depth",
            "corridor/open-space axis estimation"
        ],
        "detect": "current route's dominant free-space axis",
        "candidates": "deep floor near axis centerline within a narrow forward angular cone",
        "constraints": [
            "exclude side branches",
            "exclude backtracking",
            "exclude near-camera and wall-adjacent floor"
        ],
        "rank": "heading consistency, depth, centerline clearance",
        "arrival": "specified distance/progress or next decision area reached",
        "fallback": "choose shorter centerline point rather than changing direction",
        "waypoints": "one receding-horizon forward point"
    },
    "APPROACH_LANDMARK": {
        "perception": [
            "open-vocabulary landmark detection",
            "instance mask",
            "depth"
        ],
        "detect": "referenced landmark",
        "candidates": "floor in a safe-distance annulus on the near side of the landmark",
        "constraints": [
            "candidate remains before landmark unless side specified",
            "safe stopping distance",
            "not instance pixels"
        ],
        "rank": "text match, requested bearing relation, target-distance error, clearance",
        "arrival": "landmark has requested apparent size/distance and relation",
        "fallback": "move toward detector bearing using short floor point while retaining landmark in view",
        "waypoints": "short approach points until safe annulus becomes visible"
    },
    "STOP_WAIT": {
        "perception": [
            "open-vocabulary landmark/region detection",
            "relation parser",
            "floor segmentation",
            "depth"
        ],
        "detect": "final reference landmark/region and requested at/near/by/in-front relation",
        "candidates": "floor satisfying relation and safe offset",
        "constraints": [
            "relation must be visually verifiable",
            "zero collision overlap",
            "do not overshoot landmark"
        ],
        "rank": "relation satisfaction, detector confidence, stopping-distance error, clearance",
        "arrival": "relation and safe distance hold; then emit stop",
        "fallback": "approach conservatively while keeping landmark visible; do not stop on generic floor",
        "waypoints": "approach point followed by final stop point"
    },
    "OTHER": {
        "perception": [
            "VLM clause interpretation",
            "open-vocabulary detection",
            "floor segmentation",
            "depth"
        ],
        "detect": "entities and spatial relations explicitly mentioned in the clause",
        "candidates": "floor grounded by detected entities and parsed relations",
        "constraints": [
            "must cite detected evidence",
            "exclude backtracking and non-walkable rays"
        ],
        "rank": "evidence completeness, relation satisfaction, route continuity",
        "arrival": "VLM verifies clause-specific evidence",
        "fallback": "return no target and request another panorama or manual taxonomy extension",
        "waypoints": "clause dependent"
    }
}
