parallelExecution in Global := false

lazy val DifferentiableKernel = project.dependsOn(OpenCL, OpenCLCodeGenerator, Layer)

lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

lazy val Layer = project.dependsOn(`stateless-future`)

lazy val Memory = project

lazy val IsClosed = project

lazy val `stateless-future` = project

lazy val `stateless-future-util` = project.dependsOn(`stateless-future`)

lazy val OpenCL = project.dependsOn(IsClosed, `stateless-future`, Memory)

crossScalaVersions := Seq("2.11.8", "2.12.1")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalaOrganization in updateSbtClassifiers := (scalaOrganization in Global).value,
    scalaOrganization := "org.typelevel",
    scalacOptions += "-Yliteral-types"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
