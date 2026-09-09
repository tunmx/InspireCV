function(inspirecv_check_task_dependency_boundaries task_root)
    file(GLOB_RECURSE task_engine_files
        "${task_root}/planning/*.h"
        "${task_root}/planning/*.cc"
        "${task_root}/runtime/*.h"
        "${task_root}/runtime/*.cc"
        "${task_root}/execution/*.h"
        "${task_root}/execution/*.cc"
        "${task_root}/geometry/*.h"
        "${task_root}/geometry/*.cc"
        "${task_root}/kernels/*.h"
        "${task_root}/kernels/*.cc"
        "${task_root}/platform/*.h"
        "${task_root}/platform/*.cc")

    set(forbidden_task_dependencies
        "inspirecv/task/core/stream_task.h"
        "inspirecv/task/pipeline.h"
        "inspirecv/core/image.h"
        "inspirecv/inspirecv.h")

    foreach(source_file IN LISTS task_engine_files)
        file(READ "${source_file}" source_contents)
        foreach(forbidden_header IN LISTS forbidden_task_dependencies)
            string(FIND "${source_contents}" "${forbidden_header}" match_position)
            if(NOT match_position EQUAL -1)
                message(FATAL_ERROR
                    "Task dependency boundary violation: ${source_file} includes "
                    "${forbidden_header}. Image and compatibility facades belong "
                    "only in the Task API layer.")
            endif()
        endforeach()
    endforeach()
endfunction()

function(inspirecv_check_core_dependency_boundaries core_root)
    file(GLOB_RECURSE core_neutral_files
        "${core_root}/geometry/*.h"
        "${core_root}/geometry/*.cc"
        "${core_root}/geometry/*.cpp"
        "${core_root}/runtime/*.h"
        "${core_root}/runtime/*.cc"
        "${core_root}/runtime/*.cpp"
        "${core_root}/version/*.h"
        "${core_root}/version/*.in")

    set(forbidden_core_dependencies
        "inspirecv/backends/"
        "inspirecv/task/"
        "opencv2/")

    foreach(source_file IN LISTS core_neutral_files)
        file(READ "${source_file}" source_contents)
        foreach(forbidden_header IN LISTS forbidden_core_dependencies)
            string(FIND "${source_contents}" "${forbidden_header}" match_position)
            if(NOT match_position EQUAL -1)
                message(FATAL_ERROR
                    "Core dependency boundary violation: ${source_file} references "
                    "${forbidden_header}. Backend and Task dependencies do not "
                    "belong in neutral core code.")
            endif()
        endforeach()
    endforeach()
endfunction()

function(inspirecv_check_backend_dependency_boundaries backends_root)
    file(GLOB_RECURSE backend_files
        "${backends_root}/*.h"
        "${backends_root}/*.cc"
        "${backends_root}/*.cpp")

    foreach(source_file IN LISTS backend_files)
        file(READ "${source_file}" source_contents)
        string(FIND "${source_contents}" "inspirecv/task/" task_match)
        if(NOT task_match EQUAL -1)
            message(FATAL_ERROR
                "Backend dependency boundary violation: ${source_file} references "
                "the Task module.")
        endif()

        if(NOT source_file MATCHES "/adapter/")
            foreach(public_facade IN ITEMS image point rect size transform_matrix)
                set(forbidden_header "inspirecv/core/${public_facade}.h")
                string(FIND "${source_contents}" "${forbidden_header}" match_position)
                if(NOT match_position EQUAL -1)
                    message(FATAL_ERROR
                        "Backend engine boundary violation: ${source_file} references "
                        "${forbidden_header}. Public facade dependencies belong only "
                        "in a backend adapter.")
                endif()
            endforeach()
        endif()
    endforeach()
endfunction()
