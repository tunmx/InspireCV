function(inspirecv_check_target_contract target_name expected_sources)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR
            "InspireCV integration contract violation: target '${target_name}' is missing.")
    endif()

    get_target_property(actual_type ${target_name} TYPE)
    if(INSPIRECV_BUILD_OBJECT_LIBS)
        set(expected_type "OBJECT_LIBRARY")
    elseif(INSPIRECV_BUILD_SHARED_LIBS)
        set(expected_type "SHARED_LIBRARY")
    else()
        set(expected_type "STATIC_LIBRARY")
    endif()
    if(NOT actual_type STREQUAL expected_type)
        message(FATAL_ERROR
            "InspireCV integration contract violation: target '${target_name}' is "
            "${actual_type}, expected ${expected_type}.")
    endif()

    get_target_property(interface_include_dirs
        ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
    string(FIND "${interface_include_dirs}" "${INSPIRECV_INCLUDE_DIRS}"
        public_include_position)
    if(public_include_position EQUAL -1)
        message(FATAL_ERROR
            "InspireCV integration contract violation: '${target_name}' does not "
            "publish the public include directory to normal CMake consumers.")
    endif()

    get_target_property(actual_sources ${target_name} SOURCES)
    foreach(source_file IN LISTS actual_sources)
        string(FIND "${source_file}" "$<TARGET_OBJECTS:" nested_object_position)
        if(NOT nested_object_position EQUAL -1)
            message(FATAL_ERROR
                "InspireCV integration contract violation: '${target_name}' contains "
                "a nested object library. InspireFace consumes "
                "$<TARGET_OBJECTS:inspirecv> directly, so every implementation "
                "source must belong to that target.")
        endif()
    endforeach()

    set(unique_expected_sources ${expected_sources})
    list(LENGTH unique_expected_sources expected_source_count)
    list(REMOVE_DUPLICATES unique_expected_sources)
    list(LENGTH unique_expected_sources unique_expected_source_count)
    if(NOT expected_source_count EQUAL unique_expected_source_count)
        message(FATAL_ERROR
            "InspireCV integration contract violation: the source manifest contains "
            "duplicate translation units.")
    endif()

    list(LENGTH actual_sources actual_source_count)
    if(NOT actual_source_count EQUAL expected_source_count)
        message(FATAL_ERROR
            "InspireCV integration contract violation: target '${target_name}' owns "
            "${actual_source_count} direct sources, but the manifest has "
            "${expected_source_count}.")
    endif()
endfunction()
