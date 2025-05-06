const client = require('prom-client');

// Create a Registry to register metrics
const register = new client.Registry();

// Add default metrics (GC, memory usage, etc.)
client.collectDefaultMetrics({ register });

// Create custom metrics
const httpRequestDurationMicroseconds = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
});

const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

const sentimentAnalysisTotal = new client.Counter({
  name: 'sentiment_analysis_total',
  help: 'Total number of sentiment analyses performed',
  labelNames: ['sentiment']
});

const cacheHitRatio = new client.Gauge({
  name: 'cache_hit_ratio',
  help: 'Ratio of cache hits to total cache lookups'
});

const databaseConnectionPoolSize = new client.Gauge({
  name: 'database_connection_pool_size',
  help: 'Current size of the database connection pool'
});

const databaseQueryDurationSeconds = new client.Histogram({
  name: 'database_query_duration_seconds',
  help: 'Duration of database queries in seconds',
  labelNames: ['query_type'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
});

// Register all metrics
register.registerMetric(httpRequestDurationMicroseconds);
register.registerMetric(httpRequestsTotal);
register.registerMetric(sentimentAnalysisTotal);
register.registerMetric(cacheHitRatio);
register.registerMetric(databaseConnectionPoolSize);
register.registerMetric(databaseQueryDurationSeconds);

module.exports = {
  register,
  metrics: {
    httpRequestDurationMicroseconds,
    httpRequestsTotal,
    sentimentAnalysisTotal,
    cacheHitRatio,
    databaseConnectionPoolSize,
    databaseQueryDurationSeconds
  }
};