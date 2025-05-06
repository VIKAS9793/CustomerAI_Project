const { metrics } = require('../metrics');

/**
 * Middleware to collect HTTP metrics
 */
function metricsMiddleware(req, res, next) {
  const start = Date.now();
  
  // Record end time and metrics on response finish
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000; // Convert to seconds
    
    // Record request duration
    metrics.httpRequestDurationMicroseconds
      .labels(req.method, req.route?.path || req.path, res.statusCode.toString())
      .observe(duration);
    
    // Increment request counter
    metrics.httpRequestsTotal
      .labels(req.method, req.route?.path || req.path, res.statusCode.toString())
      .inc();
  });
  
  next();
}

module.exports = metricsMiddleware;