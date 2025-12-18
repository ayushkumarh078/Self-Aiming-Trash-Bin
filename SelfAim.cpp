#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <pigpio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "nlohmann/json.hpp"

using namespace std;
using namespace cv;
using namespace dnn;
using json = nlohmann::json;

// ---------------- Configuration Structure ----------------
struct Config {
    string targetClass;
    float confidenceThreshold;
    int centerTolerance;
    int horizontalTolerance;
    vector<int> relayPins;
    int pulseDurationMs;
    int moveDelayMs;
    int skipFrames = 2;  // Process every Nth frame for better FPS
};

// ---------------- Load Configuration ----------------
Config loadConfig(const string &filename) {
    ifstream f(filename);
    if (!f.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    json j;
    f >> j;

    Config cfg;
    cfg.targetClass = j["target_class"];
    cfg.confidenceThreshold = j["confidence_threshold"];
    cfg.centerTolerance = j["center_tolerance"];
    cfg.horizontalTolerance = j["horizontal_tolerance"];
    for (auto &p : j["relay_pins"]) cfg.relayPins.push_back(p);
    cfg.pulseDurationMs = j["pulse_duration_ms"];
    cfg.moveDelayMs = j["move_delay_ms"];
    if (j.contains("skip_frames")) cfg.skipFrames = j["skip_frames"];

    return cfg;
}

// ---------------- Relay Control Helpers ----------------
void activateRelay(int pin, int durationMs) {
    gpioWrite(pin, 0);  // Active LOW → ON
    gpioDelay(durationMs * 1000);
    gpioWrite(pin, 1);  // OFF
}

// Non-blocking relay activation
void activateRelayNonBlocking(int pin) {
    gpioWrite(pin, 0);  // Active LOW → ON
}

void deactivateRelay(int pin) {
    gpioWrite(pin, 1);  // OFF
}

// ---------------- Main Program ----------------
int main() {
    Config cfg = loadConfig("config.json");

    // Initialize pigpio
    if (gpioInitialise() < 0) {
        cerr << "❌ Failed to initialize pigpio!" << endl;
        return 1;
    }

    // Setup relay pins
    for (int pin : cfg.relayPins) {
        gpioSetMode(pin, PI_OUTPUT);
        gpioWrite(pin, 1);
    }

    // Load MobileNet-SSD model
    Net net = readNetFromCaffe(
        "models/MobileNetSSD_deploy.prototxt",
        "models/MobileNetSSD_deploy.caffemodel"
    );
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    vector<string> classNames = {
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    cout << "✅ Model loaded. Tracking: " << cfg.targetClass << endl;

    // Initialize camera with optimized settings
    VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "❌ Cannot open webcam!" << endl;
        gpioTerminate();
        return -1;
    }
    
    // Set lower resolution for faster processing
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_BUFFERSIZE, 1); // Reduce buffer to minimize latency

    Mat frame, blob;
    int targetIndex = find(classNames.begin(), classNames.end(), cfg.targetClass) - classNames.begin();
    
    // Pre-calculate frame center (constant values)
    int cx_frame, cy_frame;

    bool firstFrame = true;
    const Scalar meanVal(127.5, 127.5, 127.5);
    const Size blobSize(300, 300);
    
    int frameCounter = 0;
    Mat detections;
    
    // FPS tracking
    auto lastTime = chrono::high_resolution_clock::now();
    int fpsCounter = 0;
    double fps = 0.0;
    
    // Non-blocking relay timing
    vector<chrono::high_resolution_clock::time_point> relayStartTimes(cfg.relayPins.size());
    vector<bool> relayActive(cfg.relayPins.size(), false);
    
    // Tracking improvements
    int lastTargetX = -1, lastTargetY = -1;
    float lastConfidence = 0.0f;
    int lostFrames = 0;
    const int maxLostFrames = 10; // Stop tracking after this many frames without detection
    
    // Smoothing for stable tracking
    const float smoothingFactor = 0.7f; // Higher = more smoothing (0-1)
    
    while (true) {
        auto loopStart = chrono::high_resolution_clock::now();
        
        // Check and deactivate relays that have been on long enough (non-blocking)
        for (size_t i = 0; i < cfg.relayPins.size(); i++) {
            if (relayActive[i]) {
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                    loopStart - relayStartTimes[i]).count();
                if (elapsed >= cfg.pulseDurationMs) {
                    deactivateRelay(cfg.relayPins[i]);
                    relayActive[i] = false;
                }
            }
        }
        
        cap >> frame;
        if (frame.empty()) break;

        // Calculate frame center only once
        if (firstFrame) {
            cx_frame = frame.cols / 2;
            cy_frame = frame.rows / 2;
            firstFrame = false;
        }

        frameCounter++;
        
        // Skip frames for better FPS (process every Nth frame)
        bool processFrame = (frameCounter % (cfg.skipFrames + 1) == 0);
        
        if (processFrame) {
            // Prepare input blob (reuse blob Mat)
            blobFromImage(frame, blob, 0.007843, blobSize, meanVal, false);
            net.setInput(blob);
            detections = net.forward();
        }
        
        // Calculate FPS
        fpsCounter++;
        if (fpsCounter >= 30) {
            auto currentTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(currentTime - lastTime).count();
            fps = (fpsCounter * 1000.0) / duration;
            lastTime = currentTime;
            fpsCounter = 0;
        }

        // Only process detections if we ran inference this frame
        if (processFrame && !detections.empty()) {
            // Direct pointer access for faster iteration
            float* data = (float*)detections.ptr<float>();
            int numDetections = detections.size[2];
            int detectionSize = detections.size[3];

            // Find best target (highest confidence or closest to last position)
            bool targetFound = false;
            int bestCx = -1, bestCy = -1;
            int bestX1 = 0, bestY1 = 0, bestX2 = 0, bestY2 = 0;
            float bestConfidence = 0.0f;
            float bestScore = 0.0f;
            
            const int frameCols = frame.cols;
            const int frameRows = frame.rows;
            
            for (int i = 0; i < numDetections; i++) {
                float* detection = data + i * detectionSize;
                int classId = static_cast<int>(detection[1]);
                float confidence = detection[2];

                // Early exit if confidence too low
                if (confidence <= cfg.confidenceThreshold) continue;
                if (classId != targetIndex) continue;

                int x1 = detection[3] * frameCols;
                int y1 = detection[4] * frameRows;
                int x2 = detection[5] * frameCols;
                int y2 = detection[6] * frameRows;

                int cx = (x1 + x2) >> 1;
                int cy = (y1 + y2) >> 1;
                
                // Score based on confidence and proximity to last known position
                float score = confidence;
                if (lastTargetX > 0 && lastTargetY > 0) {
                    float dx = cx - lastTargetX;
                    float dy = cy - lastTargetY;
                    float distance = sqrt(dx*dx + dy*dy);
                    float maxDistance = sqrt(frameCols*frameCols + frameRows*frameRows);
                    float proximityScore = 1.0f - (distance / maxDistance);
                    score = confidence * 0.6f + proximityScore * 0.4f; // Weighted score
                }
                
                if (score > bestScore) {
                    bestScore = score;
                    bestConfidence = confidence;
                    bestCx = cx;
                    bestCy = cy;
                    bestX1 = x1;
                    bestY1 = y1;
                    bestX2 = x2;
                    bestY2 = y2;
                    targetFound = true;
                }
            }
            
            if (targetFound) {
                lostFrames = 0;
                
                // Apply smoothing for stable tracking
                int cx, cy;
                if (lastTargetX > 0 && lastTargetY > 0) {
                    cx = smoothingFactor * lastTargetX + (1 - smoothingFactor) * bestCx;
                    cy = smoothingFactor * lastTargetY + (1 - smoothingFactor) * bestCy;
                } else {
                    cx = bestCx;
                    cy = bestCy;
                }
                
                lastTargetX = cx;
                lastTargetY = cy;
                lastConfidence = bestConfidence;
                
                // Draw bounding box
                rectangle(frame, Point(bestX1, bestY1), Point(bestX2, bestY2), 
                         Scalar(0, 255, 0), 2);
                
                // Draw target center point
                circle(frame, Point(bestCx, bestCy), 5, Scalar(0, 0, 255), -1);
                
                // Draw line from center to target
                line(frame, Point(cx_frame, cy_frame), Point(cx, cy), 
                     Scalar(255, 0, 255), 2);
                
                // Display confidence
                string label = cfg.targetClass + ": " + 
                              to_string(static_cast<int>(bestConfidence * 100)) + "%";
                putText(frame, label, Point(bestX1, bestY1 - 10), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

                // Pre-calculate tolerance boundaries
                int leftBound = cx_frame - cfg.centerTolerance;
                int rightBound = cx_frame + cfg.centerTolerance;
                int topBound = cy_frame - cfg.centerTolerance;
                int bottomBound = cy_frame + cfg.centerTolerance;

                string action = "";
                auto now = chrono::high_resolution_clock::now();

                // Non-blocking relay activation - all movements happen simultaneously
                if (cx < leftBound) {
                    activateRelayNonBlocking(cfg.relayPins[1]); // turn left
                    relayStartTimes[1] = now;
                    relayActive[1] = true;
                    action += "Left ";
                } else if (cx > rightBound) {
                    activateRelayNonBlocking(cfg.relayPins[0]); // turn right
                    relayStartTimes[0] = now;
                    relayActive[0] = true;
                    action += "Right ";
                }

                if (cy < topBound) {
                    activateRelayNonBlocking(cfg.relayPins[2]); // move backward
                    relayStartTimes[2] = now;
                    relayActive[2] = true;
                    action += "Backward ";
                } else if (cy > bottomBound) {
                    activateRelayNonBlocking(cfg.relayPins[3]); // move forward
                    relayStartTimes[3] = now;
                    relayActive[3] = true;
                    action += "Forward ";
                }

                if (action.empty()) action = "Centered";

                cout << "Target " << cfg.targetClass
                     << " (" << cx << ", " << cy << ") → "
                     << action << " [Conf: " << static_cast<int>(bestConfidence * 100) << "%]" << endl;
            } else {
                lostFrames++;
                if (lostFrames > maxLostFrames) {
                    lastTargetX = -1;
                    lastTargetY = -1;
                }
            }
        }

        // Draw crosshair at frame center
        int crosshairSize = 20;
        Scalar crosshairColor = (lastTargetX > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        
        // Horizontal line
        line(frame, Point(cx_frame - crosshairSize, cy_frame), 
             Point(cx_frame + crosshairSize, cy_frame), crosshairColor, 2);
        // Vertical line
        line(frame, Point(cx_frame, cy_frame - crosshairSize), 
             Point(cx_frame, cy_frame + crosshairSize), crosshairColor, 2);
        
        // Draw tolerance zone (dead zone)
        rectangle(frame, 
                 Point(cx_frame - cfg.centerTolerance, cy_frame - cfg.centerTolerance),
                 Point(cx_frame + cfg.centerTolerance, cy_frame + cfg.centerTolerance),
                 Scalar(255, 255, 0), 1);
        
        // Display FPS and tracking status
        if (fps > 0) {
            putText(frame, "FPS: " + to_string(static_cast<int>(fps)), 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        string status = (lastTargetX > 0) ? "TRACKING" : "SEARCHING";
        Scalar statusColor = (lastTargetX > 0) ? Scalar(0, 255, 0) : Scalar(0, 165, 255);
        putText(frame, status, Point(10, 60), 
               FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);

        // Optional: show frame (disable for max performance)
        imshow("Frame", frame);
        if (waitKey(1) == 27) break; // ESC to exit
    }

    gpioTerminate();
    cout << "✅ Program terminated cleanly." << endl;
    return 0;
}
