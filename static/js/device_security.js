/**
 * CustomerAI Device Security Collection
 *
 * This script collects device security information for validation on the server.
 * It creates a device fingerprint that's sent with API requests to validate
 * the security posture of the connecting device.
 */

class DeviceSecurityCollector {
    constructor() {
        this.fingerprint = {};
        this.deviceId = this._getOrCreateDeviceId();
    }

    /**
     * Get or create a persistent device ID
     */
    _getOrCreateDeviceId() {
        let deviceId = localStorage.getItem('customerai_device_id');

        if (!deviceId) {
            // Generate new UUID for device
            deviceId = this._generateUUID();
            localStorage.setItem('customerai_device_id', deviceId);
        }

        return deviceId;
    }

    /**
     * Generate a UUID v4
     */
    _generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    /**
     * Check for indicators of rooted/jailbroken devices
     */
    async checkRootIndicators() {
        const rootIndicators = [];

        // Check for common root/jailbreak artifacts
        try {
            // Test localStorage access (often breaks in private mode)
            const testKey = 'test_security_' + Date.now();
            localStorage.setItem(testKey, 'test');
            localStorage.removeItem(testKey);

            // Test for suspicious global variables that might indicate jailbreak
            const suspiciousGlobals = ['Cydia', 'Substrate', 'MobileSubstrate', 'frida'];

            for (const name of suspiciousGlobals) {
                if (window[name]) {
                    rootIndicators.push(`suspicious_global_${name}`);
                }
            }

            // Check for suspicious timing indicators of tampering
            const start = performance.now();
            for (let i = 0; i < 100000; i++) {
                // Simple loop to detect debuggers or code tampering
                const x = i * i;
            }
            const end = performance.now();
            const duration = end - start;

            // If timing is much slower than expected, might indicate debugger
            if (duration > 500) { // Threshold in milliseconds
                rootIndicators.push('timing_anomaly');
            }

        } catch (error) {
            rootIndicators.push('security_exception');
        }

        this.fingerprint.root_indicators = rootIndicators;

        return rootIndicators.length === 0;
    }

    /**
     * Check if device is likely an emulator
     */
    checkEmulatorIndicators() {
        const indicators = [];

        try {
            // Check screen dimensions (emulators often have specific sizes)
            const width = window.screen.width;
            const height = window.screen.height;

            // Common emulator resolutions
            const emulatorResolutions = [
                '320x480', '480x800', '720x1280', '1080x1920',
                '800x1280', '1200x1920', '1536x2048', '2048x1536'
            ];

            if (emulatorResolutions.includes(`${width}x${height}`)) {
                indicators.push('emulator_resolution');
            }

            // Check for WebView properties that might indicate emulation
            const userAgent = navigator.userAgent.toLowerCase();
            if (userAgent.includes('android sdk') || userAgent.includes('emulator') ||
                userAgent.includes('sdk_gphone') || userAgent.includes('sdk_x86')) {
                indicators.push('emulator_user_agent');
            }

            // Check for abnormal performance characteristics
            const perfEntries = performance.getEntriesByType('navigation');
            if (perfEntries.length > 0) {
                const navPerf = perfEntries[0];
                // Unusually fast or slow timing can indicate emulation
                if (navPerf.domComplete < 10 || navPerf.domComplete > 10000) {
                    indicators.push('abnormal_performance');
                }
            }

        } catch (error) {
            console.error('Error checking for emulator:', error);
        }

        this.fingerprint.is_emulator = indicators.length > 0;
        this.fingerprint.emulator_indicators = indicators;

        return indicators.length === 0;
    }

    /**
     * Check for developer mode indicators
     */
    checkDeveloperMode() {
        let developerMode = false;
        let usbDebugging = false;

        try {
            // Check for developer console
            const devtoolsOpen = /./;
            devtoolsOpen.toString = function() {
                developerMode = true;
                return '';
            };

            // Try to detect if console is open
            console.log('%c', devtoolsOpen);

            // Look for debugging flags in navigator
            if (navigator.userAgent.toLowerCase().includes('debug') ||
                navigator.userAgent.toLowerCase().includes('development')) {
                developerMode = true;
            }

        } catch (error) {
            // Error during check
        }

        this.fingerprint.developer_mode = developerMode;
        this.fingerprint.usb_debugging = usbDebugging;

        return !developerMode;
    }

    /**
     * Check for bootloader/system modifications
     * Note: This is limited in browser, but we do our best to detect anomalies
     */
    checkSystemIntegrity() {
        let customRom = false;
        let bootloaderState = 'unknown';
        let verifiedBoot = 'unknown';

        try {
            // Check for custom browser modifications that may indicate custom ROM
            const navigator_prototype = Object.getPrototypeOf(navigator);
            const navigator_props = Object.getOwnPropertyNames(navigator_prototype);

            const suspicious_props = ['rooted', 'jailbroken', 'unlocked'];
            for (const prop of suspicious_props) {
                if (navigator_props.includes(prop)) {
                    customRom = true;
                    break;
                }
            }

            // Check for unusual browser properties or values
            if (navigator.platform === 'UNKNOWN' ||
                navigator.appVersion.includes('CUSTOM') ||
                navigator.userAgent.includes('Custom')) {
                customRom = true;
            }

        } catch (error) {
            // Error during check
        }

        this.fingerprint.custom_rom = customRom;
        this.fingerprint.bootloader_state = bootloaderState;
        this.fingerprint.verified_boot = verifiedBoot;

        return !customRom;
    }

    /**
     * Check for security settings like encryption
     */
    checkSecuritySettings() {
        let encryptionStatus = 'unknown';

        try {
            // Check if crypto APIs are available (as a proxy for security features)
            if (window.crypto && window.crypto.subtle) {
                encryptionStatus = 'enabled';
            } else {
                encryptionStatus = 'disabled';
            }

            // Check for secure context
            if (window.isSecureContext) {
                this.fingerprint.secure_context = true;
            } else {
                this.fingerprint.secure_context = false;
            }

            // Check if cookies are enabled
            this.fingerprint.cookies_enabled = navigator.cookieEnabled;

            // For HTTPS
            this.fingerprint.is_https = window.location.protocol === 'https:';

        } catch (error) {
            // Error during check
        }

        this.fingerprint.encryption_status = encryptionStatus;

        return encryptionStatus === 'enabled';
    }

    /**
     * Collect all device information
     */
    async collectDeviceInfo() {
        // Platform information
        this.fingerprint.platform = navigator.platform;
        this.fingerprint.user_agent = navigator.userAgent;
        this.fingerprint.language = navigator.language;
        this.fingerprint.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        this.fingerprint.screen_width = window.screen.width;
        this.fingerprint.screen_height = window.screen.height;
        this.fingerprint.device_pixel_ratio = window.devicePixelRatio;
        this.fingerprint.color_depth = window.screen.colorDepth;

        // Check if mobile device
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        this.fingerprint.is_mobile = isMobile;

        // Browser capabilities
        this.fingerprint.local_storage = !!window.localStorage;
        this.fingerprint.session_storage = !!window.sessionStorage;
        this.fingerprint.indexed_db = !!window.indexedDB;

        // Security checks
        await this.checkRootIndicators();
        this.checkEmulatorIndicators();
        this.checkDeveloperMode();
        this.checkSystemIntegrity();
        this.checkSecuritySettings();

        // Add device ID
        this.fingerprint.device_id = this.deviceId;

        // Add timestamp
        this.fingerprint.timestamp = new Date().toISOString();

        return this.fingerprint;
    }

    /**
     * Get the security fingerprint as a string for headers
     */
    async getDeviceFingerprint() {
        await this.collectDeviceInfo();
        return JSON.stringify(this.fingerprint);
    }

    /**
     * Add the security headers to an API request
     */
    async addSecurityHeaders(headers = {}) {
        const fingerprint = await this.getDeviceFingerprint();

        return {
            ...headers,
            'X-Device-Fingerprint': fingerprint,
            'X-Device-ID': this.deviceId
        };
    }
}

// Initialize and export the collector
const deviceSecurity = new DeviceSecurityCollector();

// Add security headers to all API requests
async function secureApiRequest(url, options = {}) {
    const securityHeaders = await deviceSecurity.addSecurityHeaders(options.headers || {});

    return fetch(url, {
        ...options,
        headers: securityHeaders
    });
}

// Export the API
window.CustomerAI = window.CustomerAI || {};
window.CustomerAI.Security = {
    deviceSecurity,
    secureApiRequest
};
