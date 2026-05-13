# Security Policy

## Reporting Security Vulnerabilities

We take the security of `mindspore-tools-mcp` seriously. If you discover a security vulnerability, please follow these steps:

### 📧 Contact

**Email**: 1309848726@qq.com

### 🔒 Responsible Disclosure

Please **DO NOT** open a public GitHub Issue for security vulnerabilities.

Instead, send an email with details:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### ⏱️ Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Release**: Depending on severity

### 🛡️ Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

### 🏆 Recognition

Contributors who responsibly disclose vulnerabilities will be acknowledged (unless they prefer to remain anonymous).

---

## Security Best Practices for Users

### When Using This Tool

1. **API Keys**: Never commit API keys or credentials to the repository
2. **Data Files**: The `data/` directory contains model lists and API mappings - these are public data, not secrets
3. **MCP Configuration**: When configuring MCP clients, ensure your configuration file is not publicly accessible

### MCP Security Considerations

- This tool runs locally via MCP - no data is sent to external servers
- All model data and API mappings are fetched from official sources (MindSpore/Gitee)
- Code execution is local - ensure you review generated code before running

---

## Dependency Security

We regularly update dependencies to patch known vulnerabilities:

```bash
# Check for vulnerable dependencies
uv pip audit

# Update dependencies
uv sync --upgrade
```

---

## License

This security policy is subject to the project's MIT License.

---

*Last updated: 2026-05-14*