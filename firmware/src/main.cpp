/* A_iFace v0.1 ESP32 safety placeholder. No motor pins are configured.
 * No commands are executed. A real hardware interface needs separate review.
 */
#include <Arduino.h>
void setup() { Serial.begin(115200); Serial.println("A_iFace v0.1 PREVIEW_ONLY; MOTORS DISABLED"); }
void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read();
    Serial.println("ERR,MOTORS_DISABLED");
  }
  delay(20);
}
