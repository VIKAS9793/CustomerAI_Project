# Security Policy

## Reporting Security Issues

Please report security issues by sending an email to security@customerai.com. Please do not create public GitHub issues for security issues.

## Security Measures

### Input Validation
- All API endpoints validate and sanitize incoming data
- Path validation for file operations
- Input sanitization for command execution

### Authentication
- API endpoints require valid API keys
- Rate limiting implemented
- IP whitelisting available

### Dependency Management
- Regular dependency updates
- Security patch application
- Security audit implementation
- Best practices updates

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Responsible Disclosure

1. Email security@customerai.com with details
2. We will acknowledge receipt within 48 hours
3. We will work on a fix
4. We will coordinate disclosure with you
5. We will credit you in our changelog

## Security Contact

- Email: security@customerai.com
- PGP Key: Available upon request
