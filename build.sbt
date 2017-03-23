parallelExecution in Global := false

lazy val DifferentiableKernel =
  project.dependsOn(OpenCL, OpenCLCodeGenerator, CumulativeTape, Layer, `stateless-future-util`, Symbolic % Test)

lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

// TODO: Create a separate Tape library?
lazy val Layer = project.dependsOn(`stateless-future`, `stateless-future-scalatest` % Test)

lazy val CumulativeTape = project.dependsOn(Layer)

lazy val CheckedTape = project.dependsOn(Layer, Closeables)

lazy val Memory = project

lazy val Closeables = project.dependsOn(`stateless-future`, `stateless-future-scalatest` % Test)

// TODO: Rename to ToLiteral?
lazy val Symbolic = project.dependsOn(Layer)

lazy val `stateless-future` = project

lazy val `stateless-future-scalaz` = project.dependsOn(`stateless-future`, `stateless-future-scalatest` % Test)

lazy val `stateless-future-sde` = project.dependsOn(`stateless-future-scalaz`, `stateless-future-scalatest` % Test)

lazy val `stateless-future-scalatest` = project.dependsOn(`stateless-future-util`)

lazy val `stateless-future-util` = project.dependsOn(`stateless-future`)

lazy val OpenCL = project.dependsOn(Closeables, `stateless-future`, Memory)

lazy val LayerFactory = project.dependsOn(DifferentiableKernel)

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
