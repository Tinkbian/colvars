%module openmm_colvars
%pythonbegin %{
import openmm
%}
%{
#include "ColvarsForce.h"
#include "ColvarsTABFIntegrator.h"
#include "openmm/Context.h"
#include <exception>
%}
%include "std_string.i"
%include "std_vector.i"
%import(module="openmm.openmm") "openmm_colvars_openmm_import.i"
namespace std { %template(vector_double) vector<double>; }
%exception {
    try { $action }
    catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        SWIG_fail;
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception in OpenMM Colvars");
        SWIG_fail;
    }
}
%ignore ColvarsPlugin::ColvarsForce::hasExplicitTabfSetting;
%include "ColvarsForce.h"
%feature("shadow") ColvarsPlugin::ColvarsTABFIntegrator::ColvarsTABFIntegrator %{
    def __init__(self, stepSize, temperature, frictionCoeff, initialAlpha=1.0,
                 physicalForceGroup=0, colvarsForceGroup=1, alphaVariableName="tabf_alpha",
                 alphaDerivativeVariableName="tabf_dE_dalpha"):
        if openmm.unit.is_quantity(stepSize):
            stepSize = stepSize.value_in_unit(openmm.unit.picosecond)
        if openmm.unit.is_quantity(temperature):
            temperature = temperature.value_in_unit(openmm.unit.kelvin)
        if openmm.unit.is_quantity(frictionCoeff):
            frictionCoeff = frictionCoeff.value_in_unit(openmm.unit.picosecond**-1)
        args = (stepSize, temperature, frictionCoeff, initialAlpha, physicalForceGroup,
                colvarsForceGroup, alphaVariableName, alphaDerivativeVariableName)
        _openmm_colvars.ColvarsTABFIntegrator_swiginit(self, $action(*args))
%}
// The inherited Python wrapper handles quantities and dispatches virtually.
%ignore ColvarsPlugin::ColvarsTABFIntegrator::setStepSize;
%ignore ColvarsPlugin::ColvarsTABFIntegrator::derivativeParameterName;
%ignore ColvarsPlugin::ColvarsTABFIntegrator::isCollectingBias;
%ignore ColvarsPlugin::ColvarsTABFIntegrator::setNextAlpha;
%ignore ColvarsPlugin::ColvarsTABFIntegrator::nextAlpha;
%include "ColvarsTABFIntegrator.h"
