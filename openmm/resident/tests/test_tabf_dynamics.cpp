#include "../src/CudaTabfDynamics.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>

static void require(bool condition) {
    if (!condition) throw std::runtime_error("TABF scalar reference mismatch");
}

static void close(double observed, double expected) {
    require(std::abs(observed-expected) <= 2.e-14*std::max(1.0, std::abs(expected)));
}

int main() {
    for (auto seed : {0ULL, 709ULL, 0xfedcba9876543210ULL}) {
        TabfGaussianState rng{};
        tabfSeed(rng, seed);
        std::mt19937_64 reference(seed);
        for (int i = 0; i < 1000; ++i) require(tabfRandomWord(rng) == reference());
#ifdef __GLIBCXX__
        std::normal_distribution<double> gaussian;
        for (int i = 0; i < 1000; ++i) close(tabfGaussian(rng), gaussian(reference));
#endif
    }

    TabfExtConfig cfg{4.0, 5.0, 1.5, 0.025, 0.0, 2.0, 1, 1, 0};
    TabfExtState state{};
    auto view = tabfExtView(cfg, state, -0.4, 10.0, false);
    close(view.x, 0.0);
    close(view.systemForce, -2.0);
    close(view.springEnergy, 0.4);
    close(tabfExtAtomicForce(cfg, view, 0.1), 2.1);
    // Physical walls act on the geometric coordinate. Their force does not
    // enter the extended coordinate's kick. The lower wall reflects once.
    const double dt = 2, z = -0.125;
    const double firstKick = dt*(view.systemForce + 0.25)/cfg.mass;
    const double noise = std::sqrt(1-std::exp(-2*cfg.frictionPerFs*dt))*std::sqrt(cfg.kT/cfg.mass);
    const double afterOU = std::exp(-cfg.frictionPerFs*dt)*firstKick + noise*z;
    require(tabfExtAdvance(cfg, state, view, 0.25, 0.1, dt, z, 1) == 0);
    close(state.x, -dt*(firstKick+afterOU)/2);
    close(state.v, -afterOU/2);
    close(state.kineticEnergy, 0.5*cfg.mass*firstKick*firstKick/4);
    close(tabfExtView(cfg,state,0.1,0,true).x, 0.0);
    close(tabfExtView(cfg,state,0.1,0,false).x, state.x);
    const auto committed = state;
    require(tabfExtAdvance(cfg,state,view,0,0,dt,0,1) == 1);
    require(tabfExtAdvance(cfg,state,view,-1.e5,0,dt,0,2) == 3);
    close(state.x, committed.x);
    close(state.v, committed.v);
    require(state.lastStep == 1);

    cfg = {8.0, 0.0, 0.0, 0.0, 0.5, 1.5, 1, 1, 1};
    state = {};
    view = tabfExtView(cfg,state,1.0,2.0,false);
    close(view.systemForce, 2.0);
    close(tabfExtAtomicForce(cfg,view,99.0), 0.0);
    // Both kinds of scalar bias act directly on external alpha.
    require(tabfExtAdvance(cfg,state,view,0.25,-0.125,1.0,0.0,10) == 0);
    close(state.x, 1.0+2.125/8.0);
    close(state.v, 2.125/8.0);
    close(state.springEnergy, 0.0);
}
