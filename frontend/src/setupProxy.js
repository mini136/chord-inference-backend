// Proxy configuration for development
// This will proxy /api requests to the backend running on localhost:40150

const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    "/api",
    createProxyMiddleware({
      target: "http://localhost:40150",
      changeOrigin: true,
    }),
  );
};
