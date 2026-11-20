%module openmm
namespace OpenMM {
class Context {};
class Integrator {};
class CustomIntegrator : public Integrator {};
class Force {
public:
    virtual ~Force();
    int getForceGroup() const;
    void setForceGroup(int group);
    virtual bool usesPeriodicBoundaryConditions() const;
};
}
