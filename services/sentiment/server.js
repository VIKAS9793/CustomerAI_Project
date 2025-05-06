/**
 * CustomerAI Insights Platform - Sentiment Analysis Service
 *
 * This microservice handles sentiment analysis of customer interactions,
 * implementing caching, resilience patterns, and performance optimizations.
 */

const express = require('express');
const { Pool } = require('pg');
const Redis = require('ioredis');
const { CircuitBreaker } = require('opossum');
const { v4: uuidv4 } = require('uuid');
const { Mutex } = require('async-mutex');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Redis client for caching
const redis = new Redis(process.env.REDIS_URL || 'redis://redis:6379');

// Initialize database connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Create mutex for cache population
const cacheMutex = new Mutex();

// Parse JSON request bodies
app.use(express.json());

// Add request ID to each request
app.use((req, res, next) => {
  req.id = req.headers['x-request-id'] || uuidv4();
  next();
});

// Logging middleware
app.use((req, res, next) => {
  console.log(`[${req.id}] ${req.method} ${req.path} - ${new Date().toISOString()}`);
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'sentiment-analysis',
    timestamp: new Date().toISOString()
  });
});

/**
 * Analyze sentiment of a single text
 * POST /sentiment
 */
app.post('/sentiment', async (req, res) => {
  const { text, use_ai = true } = req.body;

  if (!text) {
    return res.status(400).json({
      error: true,
      status_code: 400,
      message: 'Validation error',
      details: {
        text: 'Field is required'
      },
      timestamp: new Date().toISOString()
    });
  }

  try {
    // Generate cache key based on text content and AI flag
    const normalizedText = text.trim().toLowerCase();
    const cacheKey = `sentiment:${Buffer.from(normalizedText).toString('base64')}:${use_ai}`;

    // Try to get result from cache
    const cachedResult = await redis.get(cacheKey);
    if (cachedResult) {
      console.log(`[${req.id}] Cache hit for sentiment analysis`);
      return res.status(200).json(JSON.parse(cachedResult));
    }

    // Cache miss - perform sentiment analysis
    console.log(`[${req.id}] Cache miss for sentiment analysis`);

    // Use mutex to prevent cache stampede for popular queries
    const release = await cacheMutex.acquire();
    try {
      // Double-check cache after acquiring mutex
      const cachedResultRetry = await redis.get(cacheKey);
      if (cachedResultRetry) {
        console.log(`[${req.id}] Cache hit after mutex acquisition`);
        return res.status(200).json(JSON.parse(cachedResultRetry));
      }

      // Perform sentiment analysis
      const result = await analyzeSentiment(text, use_ai);

      // Cache the result (TTL: 1 day)
      await redis.setex(cacheKey, 86400, JSON.stringify(result));

      return res.status(200).json(result);
    } finally {
      release();
    }
  } catch (error) {
    console.error(`[${req.id}] Error analyzing sentiment:`, error);

    return res.status(500).json({
      error: true,
      status_code: 500,
      message: 'Error analyzing sentiment',
      correlation_id: req.id,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Perform batch sentiment analysis
 * POST /batch
 */
app.post('/batch', async (req, res) => {
  const { conversations } = req.body;

  if (!conversations || !Array.isArray(conversations) || conversations.length === 0) {
    return res.status(400).json({
      error: true,
      status_code: 400,
      message: 'Validation error',
      details: {
        conversations: 'Field must be a non-empty array'
      },
      timestamp: new Date().toISOString()
    });
  }

  try {
    // Process conversations in parallel with concurrency limit
    const results = await Promise.all(
      conversations.map(async (conversation) => {
        const { id, text } = conversation;

        if (!id || !text) {
          return {
            id: id || 'unknown',
            error: 'Missing id or text field'
          };
        }

        try {
          // Generate cache key
          const normalizedText = text.trim().toLowerCase();
          const cacheKey = `sentiment:${Buffer.from(normalizedText).toString('base64')}:true`;

          // Try to get from cache
          const cachedResult = await redis.get(cacheKey);
          if (cachedResult) {
            const parsed = JSON.parse(cachedResult);
            return {
              id,
              sentiment: parsed.data.sentiment,
              positive: parsed.data.positive,
              negative: parsed.data.negative,
              neutral: parsed.data.neutral
            };
          }

          // Perform sentiment analysis
          const result = await analyzeSentiment(text, true);

          // Cache the result
          await redis.setex(cacheKey, 86400, JSON.stringify(result));

          return {
            id,
            sentiment: result.data.sentiment,
            positive: result.data.positive,
            negative: result.data.negative,
            neutral: result.data.neutral
          };
        } catch (error) {
          console.error(`[${req.id}] Error analyzing sentiment for conversation ${id}:`, error);
          return {
            id,
            error: 'Failed to analyze sentiment'
          };
        }
      })
    );

    // Count results by sentiment
    const summary = results.reduce(
      (acc, result) => {
        if (result.error) {
          acc.error_count++;
        } else if (result.sentiment === 'positive') {
          acc.positive_count++;
        } else if (result.sentiment === 'negative') {
          acc.negative_count++;
        } else {
          acc.neutral_count++;
        }
        return acc;
      },
      { positive_count: 0, negative_count: 0, neutral_count: 0, error_count: 0 }
    );

    return res.status(200).json({
      error: false,
      status_code: 200,
      data: {
        results,
        summary
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error(`[${req.id}] Error processing batch:`, error);

    return res.status(500).json({
      error: true,
      status_code: 500,
      message: 'Error processing batch',
      correlation_id: req.id,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Analyze sentiment of text
 * This is a mock implementation that would be replaced with actual NLP/ML models
 */
async function analyzeSentiment(text, useAI) {
  // Simulate processing delay
  await new Promise(resolve => setTimeout(resolve, 100));

  // Simple rule-based sentiment analysis for demonstration
  // In a real implementation, this would use NLP models or external AI services
  const positiveWords = ['happy', 'great', 'excellent', 'good', 'love', 'thank', 'resolved', 'quickly'];
  const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'slow', 'issue', 'problem', 'fail'];

  const words = text.toLowerCase().split(/\W+/);

  let positiveCount = 0;
  let negativeCount = 0;

  for (const word of words) {
    if (positiveWords.includes(word)) positiveCount++;
    if (negativeWords.includes(word)) negativeCount++;
  }

  const total = words.length;
  const positive = positiveCount / total;
  const negative = negativeCount / total;
  const neutral = 1 - positive - negative;

  let sentiment;
  if (positive > negative) {
    sentiment = 'positive';
  } else if (negative > positive) {
    sentiment = 'negative';
  } else {
    sentiment = 'neutral';
  }

  // Enhanced analysis with AI (simulated)
  let analysis = {};
  if (useAI) {
    if (sentiment === 'positive') {
      analysis = {
        satisfaction_score: Math.round(7 + 3 * positive),
        key_positives: extractKeyPhrases(text, positiveWords)
      };
    } else if (sentiment === 'negative') {
      analysis = {
        satisfaction_score: Math.round(3 - 3 * negative),
        key_negatives: extractKeyPhrases(text, negativeWords),
        urgency: negative > 0.3 ? 'high' : 'medium'
      };
    } else {
      analysis = {
        satisfaction_score: 5
      };
    }
  }

  // Store result in database (async, don't wait for completion)
  storeAnalysisResult(text, sentiment, positive, negative, neutral, analysis).catch(err => {
    console.error('Error storing analysis result:', err);
  });

  return {
    error: false,
    status_code: 200,
    data: {
      sentiment,
      positive: parseFloat(positive.toFixed(2)),
      negative: parseFloat(negative.toFixed(2)),
      neutral: parseFloat(neutral.toFixed(2)),
      analysis
    },
    timestamp: new Date().toISOString()
  };
}

/**
 * Extract key phrases from text based on sentiment words
 */
function extractKeyPhrases(text, sentimentWords) {
  const phrases = [];
  const words = text.toLowerCase().split(/\W+/);

  for (let i = 0; i < words.length; i++) {
    if (sentimentWords.includes(words[i])) {
      // Extract 3-word phrase around the sentiment word
      const start = Math.max(0, i - 1);
      const end = Math.min(words.length, i + 2);
      const phrase = words.slice(start, end).join(' ');
      phrases.push(phrase);
    }
  }

  // Return unique phrases, up to 3
  return [...new Set(phrases)].slice(0, 3);
}

/**
 * Store analysis result in database
 */
async function storeAnalysisResult(text, sentiment, positive, negative, neutral, analysis) {
  try {
    const client = await pool.connect();
    try {
      await client.query(
        'INSERT INTO sentiment_analysis (text, sentiment, positive_score, negative_score, neutral_score, analysis, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)',
        [text, sentiment, positive, negative, neutral, JSON.stringify(analysis), new Date()]
      );
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('Database error:', error);
    // In a real implementation, we would use a retry mechanism or queue failed writes
  }
}

// Start the server
app.listen(PORT, () => {
  console.log(`Sentiment Analysis Service running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

async function shutdown() {
  console.log('Shutting down sentiment analysis service...');

  // Close database pool
  await pool.end();

  // Close Redis connection
  await redis.quit();

  process.exit(0);
}

// Circuit breaker configuration
const breaker = new CircuitBreaker(async (text) => {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'INSERT INTO sentiment_analysis (text, sentiment, positive_score, negative_score, neutral_score, analysis) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *',
            [text.input, text.sentiment, text.scores.positive, text.scores.negative, text.scores.neutral, text.analysis]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}, {
    timeout: 3000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
});

// Graceful shutdown handling
process.on('SIGTERM', async () => {
    await pool.end();
    await redis.quit();
    process.exit(0);
});

// ... existing code ...
