#ifndef OPENMM_RESIDENT_CUDA_TABF_VARIABLES_H
#define OPENMM_RESIDENT_CUDA_TABF_VARIABLES_H
#include "colvarmodule.h"
#include <string>
#include <vector>

class colvarproxy_openmm;
class colvar;
namespace ColvarsPlugin {
struct TabfVariable {
    int atom1 = -1, atom2 = -1, minimumImage = 0, extended = 0, axis = -1;
    double lower = 0, upper = 0, kLower = 0, kUpper = 0;
};
bool configureTabfVariables(colvarproxy_openmm& proxy, std::vector<TabfVariable>& vars, std::string& why);
void publishTabfMonitorsAndWalls(colvarproxy_openmm& proxy, const std::vector<TabfVariable>& vars,
                                const std::vector<double>& actual, const std::vector<double>& previousActual,
                                double dtFs, cvm::step_number relativeStep);
void publishTabfDiagnostic(colvar& cv, double actual, double reported);
}
#endif
