import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 50,
  duration: "10s",
  thresholds: {
    http_req_failed: ["rate<0.01"],       // <1% errors
    http_req_duration: ["p(95)<500"],     // 95% < 500ms
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

export default function () {
  const res = http.get(`${BASE_URL}/health`);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(0.1);
}
