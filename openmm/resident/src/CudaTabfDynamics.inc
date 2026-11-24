// Shared C++/NVRTC definitions. TABF_BODY preserves each declaration verbatim
// on the host and stringizes it for the runtime CUDA compiler.
TABF_BODY(
struct TabfExtConfig {
    double mass, spring, kT, frictionPerFs, lower, upper;
    int reflectLower, reflectUpper, external;
};
struct TabfExtState {
    double x, v, previousX, previousV;
    long long lastStep;
    int initialized;
    double actual, actualVelocity, systemForce, auxBias, actualBias;
    double springEnergy, kineticEnergy, noiseDt, noiseSigma;
};
struct TabfExtView {
    double x, v, systemForce, springEnergy, actual;
};
)

// actual and all forces use native Colvars units. Replay selects the held
// pre-integration state of an already committed endpoint. A reporter queries
// the current held state. Neither evaluation mutates coordinates or the RNG.
TABF_BODY(
TABF_INLINE TabfExtView tabfExtView(const TabfExtConfig& c, const TabfExtState& s,
                                    double actual, double alphaSystemForce, bool replayPrevious) {
    double x = actual;
    double v = 0;
    if (s.initialized) {
        x = replayPrevious ? s.previousX : s.x;
        v = replayPrevious ? s.previousV : s.v;
    }
    if (c.reflectLower && x < c.lower) x = c.lower;
    if (c.reflectUpper && x > c.upper) x = c.upper;
    const double delta = x - actual;
    const double force = c.external ? alphaSystemForce : -c.spring*delta;
    const double springEnergy = c.external ? 0.0 : 0.5*c.spring*delta*delta;
    return {x, v, force, springEnergy, actual};
}

TABF_INLINE double tabfExtAtomicForce(const TabfExtConfig& c, const TabfExtView& view,
                                    double actualBias) {
    return c.external ? 0.0 : -view.systemForce + actualBias;
}
)

// Return 0 on success. On an invalid clock (1), invalid parameters or
// nonfinite result (2), or a failed reflection/invalid alpha (3), the caller
// discards the staged endpoint. No field is changed until validation passes.
TABF_BODY(
TABF_INLINE int tabfExtAdvance(const TabfExtConfig& c, TabfExtState& s, const TabfExtView& view,
                              double auxBias, double actualBias, double dtFs, double gaussian,
                              long long endpoint) {
    if (endpoint < 0 || (s.initialized && (s.lastStep < 0 || endpoint <= s.lastStep ||
                                           endpoint - s.lastStep != 1))) return 1;
    if (!(dtFs > 0.0) || !TABF_MATH(isfinite)(dtFs) || !(c.mass > 0.0) ||
        !TABF_MATH(isfinite)(c.mass) || !(c.frictionPerFs >= 0.0) ||
        !TABF_MATH(isfinite)(c.frictionPerFs) || !TABF_MATH(isfinite)(view.actual) ||
        !TABF_MATH(isfinite)(view.springEnergy) || !TABF_MATH(isfinite)(actualBias)) return 2;
    const double drivenBias = c.external ? auxBias + actualBias : auxBias;
    const double halfKick = 0.5*dtFs*(drivenBias + view.systemForce)/c.mass;
    double v = view.v + halfKick;
    const double kineticEnergy = 0.5*c.mass*v*v;
    v += halfKick;
    double x = view.x + 0.5*dtFs*v;
    double sigma = s.noiseSigma;
    if (c.frictionPerFs != 0.0) {
        if (dtFs != s.noiseDt) {
            sigma = TABF_MATH(sqrt)(-TABF_MATH(expm1)(-2*c.frictionPerFs*dtFs)*c.mass*c.kT);
        }
        v = TABF_MATH(exp)(-c.frictionPerFs*dtFs)*v + sigma*gaussian/c.mass;
    }
    x += 0.5*dtFs*v;
    double overshoot = 0.0;
    if (c.reflectLower && x < c.lower) overshoot = x - c.lower;
    else if (c.reflectUpper && x > c.upper) overshoot = x - c.upper;
    if (overshoot != 0.0) {
        x -= 2.0*overshoot;
        v = -0.5*(view.v + v);
    }
    if (!TABF_MATH(isfinite)(x) || !TABF_MATH(isfinite)(v) || !TABF_MATH(isfinite)(kineticEnergy))
        return 2;
    if ((c.reflectLower && x < c.lower) || (c.reflectUpper && x > c.upper) ||
        (c.external && x <= 0.0)) return 3;
    s.actualVelocity = s.initialized ? (view.actual - (c.external ? s.x : s.actual))/dtFs : 0.0;
    s.previousX = view.x;
    s.previousV = view.v;
    s.actual = view.actual;
    s.x = x;
    s.v = v;
    s.systemForce = view.systemForce;
    s.auxBias = auxBias;
    s.actualBias = actualBias;
    s.springEnergy = view.springEnergy;
    s.kineticEnergy = kineticEnergy;
    s.noiseDt = dtFs;
    s.noiseSigma = sigma;
    s.lastStep = endpoint;
    s.initialized = 1;
    return 0;
}
)

// mt19937_64 with the libstdc++ polar normal_distribution ordering used by
// the Linux OpenMM proxy. The spare deviate and engine cursor are checkpointed.
TABF_BODY(
struct TabfGaussianState {
    unsigned long long words[312];
    int index, hasSpare;
    double spare;
};

TABF_INLINE void tabfSeed(TabfGaussianState& s, unsigned long long seed) {
    s.words[0] = seed;
    for (int i = 1; i < 312; ++i) {
        const unsigned long long prev = s.words[i-1];
        s.words[i] = 6364136223846793005ULL*(prev ^ (prev >> 62)) + i;
    }
    s.index = 312;
    s.hasSpare = 0;
    s.spare = 0;
}

TABF_INLINE unsigned long long tabfRandomWord(TabfGaussianState& s) {
    if (s.index == 312) {
        for (int i = 0; i < 312; ++i) {
            const unsigned long long y = (s.words[i] & 0xffffffff80000000ULL) |
                                         (s.words[(i+1)%312] & 0x7fffffffULL);
            s.words[i] = s.words[(i+156)%312] ^ (y >> 1) ^
                         ((y & 1ULL) ? 0xb5026f5aa96619e9ULL : 0ULL);
        }
        s.index = 0;
    }
    unsigned long long y = s.words[s.index++];
    y ^= (y >> 29) & 0x5555555555555555ULL;
    y ^= (y << 17) & 0x71d67fffeda60000ULL;
    y ^= (y << 37) & 0xfff7eee000000000ULL;
    return y ^ (y >> 43);
}

TABF_INLINE double tabfUniform(TabfGaussianState& s) {
    const double value = double(tabfRandomWord(s))*(1.0/18446744073709551616.0);
    return value < 1.0 ? value : 0.99999999999999988897769753748434595763683319091796875;
}

TABF_INLINE double tabfGaussian(TabfGaussianState& s) {
    if (s.hasSpare) {
        s.hasSpare = 0;
        return s.spare;
    }
    double x, y, r2;
    do {
        x = 2.0*tabfUniform(s) - 1.0;
        y = 2.0*tabfUniform(s) - 1.0;
        r2 = x*x + y*y;
    } while (r2 > 1.0 || r2 == 0.0);
    const double scale = TABF_MATH(sqrt)(-2.0*TABF_MATH(log)(r2)/r2);
    s.spare = x*scale;
    s.hasSpare = 1;
    return y*scale;
}
)
