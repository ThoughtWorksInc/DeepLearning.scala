sbtPlugin := true

libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.4"

publishArtifact := {
  if (scalaBinaryVersion.value == "2.10") {
    true
  } else {
    false
  }
}

skip := {
  if (scalaBinaryVersion.value == "2.10") {
    false
  } else {
    true
  }
}
