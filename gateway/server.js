/**
 * CustomerAI Insights Platform - API Gateway
 *
 * This gateway routes requests to the appropriate microservices,
 * handles authentication, rate limiting, and other cross-cutting concerns.
 */

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const { v4: uuidv4 } = require('uuid');
const Redis = require('ioredis');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8000;

// Initialize Redis client for caching and rate limiting
const redis = new Redis(process.env.REDIS_URL || 'redis://redis:6379');

// Add request ID to each request
app.use((req, res, next) => {
  req.id = uuidv4();
  res.setHeader('X-Request-ID', req.id);
  next();
});

// Security middleware
app.use(helmet());
app.use(cors());

// Logging middleware
app.use(morgan(':method :url :status :response-time ms - :req[x-request-id]'));

// Parse JSON request bodies
app.use(express.json());

// Rate limiting middleware
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 60000, // 1 minute
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100, // limit each IP to 100 requests per windowMs
  standardHeaders: true,
  store: {
    incr: (key) => {
      return new Promise((resolve, reject) => {
        redis.incr(key)
          .then(result => {
            redis.expire(key, Math.ceil(parseInt(process.env.RATE_LIMIT_WINDOW_MS) / 1000));
            resolve(result);
          })
          .catch(err => reject(err));
      });
    },
    decrement: (key) => {
      return redis.decr(key);
    },
    resetKey: (key) => {
      return redis.del(key);
    }
  }
});

app.use(limiter);

// Authentication middleware - verify JWT for protected routes
const authenticateJWT = async (req, res, next) => {
  // Skip authentication for login endpoint and health check
  if (req.path === '/api/v1/auth/login' || req.path === '/health') {
    return next();
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      error: true,
      status_code: 401,
      message: 'Unauthorized - Missing or invalid token format',
      timestamp: new Date().toISOString()
    });
  }

  const token = authHeader.split(' ')[1];

  try {
    // Check if token is in blocklist
    const isBlocked = await redis.get(`blocklist:${token}`);
    if (isBlocked) {
      return res.status(401).json({
        error: true,
        status_code: 401,
        message: 'Unauthorized - Token has been revoked',
        timestamp: new Date().toISOString()
      });
    }

    // Forward to auth service for verification
    // In a real implementation, we would verify the token here
    // or make a request to the auth service
    req.user = { id: 'user-123', roles: ['analyst'] };
    next();
  } catch (error) {
    return res.status(401).json({
      error: true,
      status_code: 401,
      message: 'Unauthorized - Invalid token',
      timestamp: new Date().toISOString()
    });
  }
};

app.use(authenticateJWT);

// Response caching middleware
const cacheMiddleware = (duration) => {
  return async (req, res, next) => {
    // Only cache GET requests
    if (req.method !== 'GET') {
      return next();
    }

    const cacheKey = `cache:${req.originalUrl}:${JSON.stringify(req.query)}`;

    try {
      const cachedResponse = await redis.get(cacheKey);
      if (cachedResponse) {
        const parsedResponse = JSON.parse(cachedResponse);
        return res.status(200).json(parsedResponse);
      }

      // Store the original send function
      const originalSend = res.send;

      // Override the send function to cache the response
      res.send = function(body) {
        if (res.statusCode === 200) {
          redis.setex(cacheKey, duration, body);
        }
        originalSend.call(this, body);
      };

      next();
    } catch (error) {
      console.error('Cache error:', error);
      next();
    }
  };
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString()
  });
});

// Service proxy routes

// Authentication Service
app.use('/api/v1/auth', createProxyMiddleware({
  target: 'http://auth-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/auth': '/'
  }
}));

// Sentiment Analysis Service
app.use('/api/v1/analyze', createProxyMiddleware({
  target: 'http://sentiment-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/analyze': '/'
  }
}));

// Response Generation Service
app.use('/api/v1/generate', createProxyMiddleware({
  target: 'http://response-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/generate': '/'
  }
}));

// Privacy Management Service
app.use('/api/v1/privacy', createProxyMiddleware({
  target: 'http://privacy-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/privacy': '/'
  }
}));

// Fairness Analysis Service
app.use('/api/v1/fairness', createProxyMiddleware({
  target: 'http://fairness-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/fairness': '/'
  }
}));

// Human Review Service
app.use('/api/v1/review', createProxyMiddleware({
  target: 'http://review-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/review': '/'
  }
}));

// Analytics Service
app.use('/api/v1/analytics', cacheMiddleware(300), createProxyMiddleware({
  target: 'http://analytics-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/analytics': '/'
  }
}));

// AI/ML Service
app.use('/api/v1/ai', createProxyMiddleware({
  target: 'http://ai-service:3000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1/ai': '/'
  }
}));

// WebSocket proxy for real-time updates
app.use('/websocket', createProxyMiddleware({
  target: 'http://review-service:3000',
  changeOrigin: true,
  ws: true,
  pathRewrite: {
    '^/websocket': '/ws'
  }
}));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(`Error [${req.id}]:`, err);

  res.status(500).json({
    error: true,
    status_code: 500,
    message: 'Internal Server Error',
    correlation_id: req.id,
    timestamp: new Date().toISOString()
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`API Gateway running on port ${PORT}`);
});
