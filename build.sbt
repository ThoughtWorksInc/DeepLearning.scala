parallelExecution in Global := false

lazy val DifferentiableKernel = project.dependsOn(OpenCL, OpenCLCodeGenerator)

lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

lazy val Memory = project

lazy val CheckedCloseable = project

lazy val `stateless-future` = project

lazy val `stateless-future-util` = project.dependsOn(`stateless-future`)

lazy val OpenCL = project.dependsOn(CheckedCloseable, `stateless-future`, Memory)

crossScalaVersions := Seq("2.10.6", "2.11.8", "2.12.1")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
